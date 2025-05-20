#include <Interpreters/JoinExpressionActions.h>
#include <stack>
#include <Core/Block.h>
#include <boost/noncopyable.hpp>
#include <Functions/isNotDistinctFrom.h>


#include <Functions/FunctionFactory.h>
#include <Functions/FunctionsLogical.h>
#include <Functions/FunctionsComparison.h>

#include <Interpreters/ActionsDAG.h>

namespace DB
{

namespace ErrorCodes
{
    extern const int INCORRECT_DATA;
    extern const int LOGICAL_ERROR;
    extern const int UNKNOWN_IDENTIFIER;
}

std::string_view toString(JoinConditionOperator op)
{
    switch (op)
    {
        case JoinConditionOperator::Equals: return "=";
        case JoinConditionOperator::NullSafeEquals: return "<=>";
        case JoinConditionOperator::Less: return "<";
        case JoinConditionOperator::LessOrEquals: return "<=";
        case JoinConditionOperator::Greater: return ">";
        case JoinConditionOperator::GreaterOrEquals: return ">=";
        case JoinConditionOperator::And: return "AND";
        case JoinConditionOperator::Or: return "OR";
        case JoinConditionOperator::Unknown: break;
    }
    throw Exception(ErrorCodes::LOGICAL_ERROR, "Illegal value for JoinConditionOperator: {}", static_cast<Int32>(op));
}

struct JoinExpressionActions::Data : boost::noncopyable
{
    using NodeToSourceMapping = std::unordered_map<NodeRawPtr, BaseRelsSet>;
    Data(ActionsDAG && actions_dag_, NodeToSourceMapping && expression_sources_) : actions_dag(std::move(actions_dag_)), expression_sources(std::move(expression_sources_)) {}

    ActionsDAG actions_dag;
    NodeToSourceMapping expression_sources;
};


JoinExpressionActions::JoinExpressionActions(const Block & left_header, const Block & right_header)
{
    Data::NodeToSourceMapping expression_sources;
    ActionsDAG actions_dag;

    for (const auto & column : left_header)
    {
        const auto * node = &actions_dag.addInput(column.name, column.type);
        expression_sources[node] = BaseRelsSet(JoinTableSide::Left);
    }

    for (const auto & column : right_header)
    {
        const auto * node = &actions_dag.addInput(column.name, column.type);
        expression_sources[node] = BaseRelsSet(JoinTableSide::Right);
    }

    data = std::make_shared<Data>(std::move(actions_dag), std::move(expression_sources));
}

JoinExpressionActions::JoinExpressionActions(const Block & left_header, const Block & right_header, ActionsDAG && actions_dag_)
{
    Data::NodeToSourceMapping expression_sources;


    const auto & input_nodes = actions_dag_.getInputs();
    if (input_nodes.size() != left_header.columns() + right_header.columns())
        throw Exception(ErrorCodes::INCORRECT_DATA, "Input nodes size mismatch in dag: {}, expected: [{}], [{}]",
                        actions_dag_.dumpDAG(), left_header.dumpNames(), right_header.dumpNames());

    for (size_t i = 0; i < input_nodes.size(); ++i)
    {
        BaseRelsSet rels;
        if (input_nodes[i]->type != ActionsDAG::ActionType::INPUT)
            throw Exception(ErrorCodes::INCORRECT_DATA, "Input node {} is not INPUT in dag: {}",
                            i, actions_dag_.dumpDAG());
        rels.set(i < left_header.columns() ? 0 : 1);
        expression_sources[input_nodes[i]] = rels;
    }

    data = std::make_shared<Data>(std::move(actions_dag_), std::move(expression_sources));
}


using NodeRawPtr = JoinExpressionActions::NodeRawPtr;

BaseRelsSet getExpressionSourcesImpl(std::unordered_map<NodeRawPtr, BaseRelsSet> & expression_sources, const JoinActionRef & action)
{
    auto node = action.getNode();
    if (auto it = expression_sources.find(node); it != expression_sources.end())
        return it->second;

    std::stack<std::pair<NodeRawPtr, size_t>> stack;
    stack.push({node, 0});

    while (!stack.empty())
    {
        auto & [current, child_idx] = stack.top();

        if (expression_sources.contains(current))
        {
            stack.pop();
            continue;
        }

        if (current->type == ActionsDAG::ActionType::INPUT)
            throw Exception(ErrorCodes::LOGICAL_ERROR,
                "Unknown input node {} in expression sources", current->result_name);

        if (child_idx >= current->children.size())
        {
            BaseRelsSet sources;
            for (const auto & child : current->children)
                sources |= expression_sources.at(child);

            expression_sources[current] = sources;
            stack.pop();
            continue;
        }

        auto child = current->children[child_idx];
        child_idx++;

        if (!expression_sources.contains(child))
            stack.push({child, 0});
    }
    return expression_sources.at(node);
}

SafeSharedPtr<ActionsDAG> JoinExpressionActions::getActionsDAG() const
{
    return std::shared_ptr<ActionsDAG>(data, &data->actions_dag);
}

JoinActionRef::JoinActionRef(NodeRawPtr node_, std::shared_ptr<JoinExpressionActions::Data> data_)
    : actions_dag(&data_->actions_dag)
    , column_name(node_->result_name)
    , data(data_)
{
}

JoinActionRef::JoinActionRef(NodeRawPtr node_, JoinExpressionActions & expression_actions_)
    : JoinActionRef(node_, expression_actions_.data)
{
}

// void JoinActionRef::serialize(WriteBuffer & out) const
// {
//     writeStringBinary(column_name, out);
// }

// JoinActionRef JoinActionRef::deserialize(ReadBuffer & in, const ActionsDAG * actions_dag_)
// {
//     String column_name;
//     readStringBinary(column_name, in);
//     const auto * node = actions_dag_->tryFindInOutputs(column_name);
//     if (!node)
//         throw Exception(ErrorCodes::INCORRECT_DATA, "Cannot find column {} in actions DAG:\n{}",
//             column_name, actions_dag_->dumpDAG());
//     return JoinActionRef(node, actions_dag_);
// }

const ActionsDAG::Node * JoinActionRef::getNode() const
{
    const auto * node = actions_dag ? actions_dag->tryFindInOutputs(column_name) : nullptr;
    if (!node)
        throw Exception(ErrorCodes::LOGICAL_ERROR, "Cannot find column {} in actions DAG:\n{}",
            column_name, actions_dag ? actions_dag->dumpDAG() : "nullptr");
    return node;
}

ColumnWithTypeAndName JoinActionRef::getColumn() const
{
    const auto * node = getNode();
    return {node->column, node->result_type, column_name};
}

const String & JoinActionRef::getColumnName() const
{
    return column_name;
}

DataTypePtr JoinActionRef::getType() const
{
    return getNode()->result_type;
}

std::vector<JoinActionRef> JoinExpressionActions::addNodes(ActionsDAG::NodeRawConstPtrs nodes)
{
    auto second_dag = ActionsDAG::cloneSubDAG(nodes, false);
    ActionsDAG::NodeRawConstPtrs second_dag_outputs;
    data->actions_dag.mergeNodes(std::move(second_dag), &second_dag_outputs);
    return std::ranges::to<std::vector>(second_dag_outputs | std::views::transform([this](const auto * node) { return JoinActionRef(node, data); }));

    // auto & actions_dag = getActionsDAG();
    // auto & outputs = actions_dag->getOutputs();
    // if (!std::ranges::contains(outputs, node))
    //     outputs.push_back(node);
    // return JoinActionRef(node, data);
}

JoinActionRef JoinExpressionActions::findNode(const String & column_name, bool is_input) const
{
    const auto & nodes = is_input ? data->actions_dag.getInputs() : data->actions_dag.getOutputs();
    for (const auto & node : nodes)
        if (node->result_name == column_name)
            return JoinActionRef(node, data);

    throw Exception(ErrorCodes::UNKNOWN_IDENTIFIER, "Cannot find column {} in actions DAG {}:\n{}",
        column_name, is_input ? "input" : "output", data->actions_dag.dumpDAG());
}


ActionsDAG JoinExpressionActions::getSubDAG(JoinActionRef action)
{
    return getSubDAG(std::views::single(action));
}

JoinExpressionActions JoinExpressionActions::clone(std::vector<JoinActionRef> & nodes) const
{

    ActionsDAG::NodePtrMap node_map;
    auto actions_dag = getActionsDAG()->clone(node_map);
    JoinExpressionActions::Data::NodeToSourceMapping new_expression_sources;
    for (const auto & [node, source] : data->expression_sources)
    {
        auto it = node_map.find(node);
        if (it == node_map.end())
            throw Exception(ErrorCodes::LOGICAL_ERROR, "Cannot find node {} in node map", node->result_name);
        new_expression_sources[it->second] = source;
    }

    auto result_data = std::make_shared<Data>(std::move(actions_dag), std::move(new_expression_sources));
    for (auto & node : nodes)
    {
        auto it = node_map.find(node.getNode());
        if (it == node_map.end())
            throw Exception(ErrorCodes::LOGICAL_ERROR, "Cannot find node in node map");
        node = JoinActionRef(it->second, result_data);
    }
    return JoinExpressionActions(std::move(result_data));
}


BaseRelsSet JoinActionRef::getExpressionSources() const
{
    return getExpressionSourcesImpl(getData()->expression_sources, *this);
}

std::shared_ptr<JoinExpressionActions::Data> JoinActionRef::getData() const
{
    auto data_ = data.lock();
    if (!data_)
        throw Exception(ErrorCodes::LOGICAL_ERROR, "Dangling JoinActionRef");
    return data_;
}

std::string operatorToFunctionName(JoinConditionOperator op)
{
    switch (op)
    {
        case JoinConditionOperator::And: return FunctionAnd::name;
        case JoinConditionOperator::Or: return FunctionOr::name;
        case JoinConditionOperator::NullSafeEquals: return FunctionIsNotDistinctFrom::name;
        case JoinConditionOperator::Equals: return NameEquals::name;
        case JoinConditionOperator::Less: return NameLess::name;
        case JoinConditionOperator::LessOrEquals: return NameLessOrEquals::name;
        case JoinConditionOperator::Greater: return NameGreater::name;
        case JoinConditionOperator::GreaterOrEquals: return NameGreaterOrEquals::name;
        case JoinConditionOperator::Unknown: break;
    }
    throw Exception(ErrorCodes::LOGICAL_ERROR, "Illegal value for JoinConditionOperator: {}", static_cast<Int32>(op));
}

JoinConditionOperator functionNameToOperator(std::string_view name)
{
    using UnderlyingType = std::underlying_type_t<JoinConditionOperator>;
    for (UnderlyingType i = 0; i < static_cast<UnderlyingType>(JoinConditionOperator::Unknown); ++i)
    {
        if (operatorToFunctionName(static_cast<JoinConditionOperator>(i)) == name)
            return static_cast<JoinConditionOperator>(i);
    }
    return JoinConditionOperator::Unknown;
}

JoinConditionOperator JoinActionRef::getFunction() const
{
    const auto * node = getNode();
    if (node->type != ActionsDAG::ActionType::FUNCTION)
        return JoinConditionOperator::Unknown;
    const auto & function_name = node->function ? node->function->getName() : "";
    return functionNameToOperator(function_name);
}

std::vector<JoinActionRef> JoinActionRef::getArguments(bool recursive) const
{
    UNUSED(recursive);
    const auto * node = getNode();
    std::vector<JoinActionRef> arguments;
    auto data_ = getData();
    for (const auto & child : node->children)
        arguments.emplace_back(child, data_);
    return arguments;
}

JoinConditionOperator JoinActionRef::asFunction(std::initializer_list<std::reference_wrapper<JoinActionRef>> operands) const
{
    const auto * node = getNode();
    if (node->type != ActionsDAG::ActionType::FUNCTION
     || node->children.size() != operands.size())
        return JoinConditionOperator::Unknown;
    size_t i = 0;
    auto data_ = getData();
    for (auto op : operands)
        op.get() = JoinActionRef(node->children[i++], data_);
    return getFunction();
}


}
