#include <Processors/QueryPlan/JoinStepLogical.h>

#include <Common/JSONBuilder.h>
#include <Common/safe_cast.h>
#include <Common/typeid_cast.h>

#include <Core/Joins.h>
#include <Core/Settings.h>

#include <DataTypes/DataTypesNumber.h>

#include <Functions/FunctionFactory.h>
#include <Functions/FunctionsComparison.h>
#include <Functions/FunctionsLogical.h>
#include <Functions/IFunctionAdaptors.h>
#include <Functions/isNotDistinctFrom.h>
#include <Functions/IsOperation.h>
#include <Functions/tuple.h>

#include <Interpreters/ActionsDAG.h>
#include <Interpreters/Context.h>
#include <Interpreters/ExpressionActions.h>
#include <Interpreters/FullSortingMergeJoin.h>
#include <Interpreters/HashJoin/HashJoin.h>
#include <Interpreters/IJoin.h>
#include <Interpreters/JoinExpressionActions.h>
#include <Interpreters/JoinInfo.h>
#include <Interpreters/JoinUtils.h>
#include <Interpreters/PasteJoin.h>
#include <Interpreters/TableJoin.h>

#include <IO/Operators.h>

#include <Planner/PlannerJoins.h>

#include <Processors/QueryPlan/CreateSetAndFilterOnTheFlyStep.h>
#include <Processors/QueryPlan/JoinStep.h>
#include <Processors/QueryPlan/Optimizations/QueryPlanOptimizationSettings.h>
#include <Processors/QueryPlan/Optimizations/Utils.h>
#include <Processors/QueryPlan/QueryPlan.h>
#include <Processors/QueryPlan/QueryPlanSerializationSettings.h>
#include <Processors/QueryPlan/QueryPlanStepRegistry.h>
#include <Processors/QueryPlan/Serialization.h>
#include <Processors/Transforms/JoiningTransform.h>

#include <QueryPipeline/QueryPipelineBuilder.h>

#include <Storages/StorageJoin.h>

#include <Processors/QueryPlan/Optimizations/joinOrder.h>
#include <algorithm>
#include <ranges>
#include <stack>

namespace DB
{

namespace Setting
{
    extern const SettingsJoinAlgorithm join_algorithm;
    extern const SettingsBool join_any_take_last_row;
    extern const SettingsUInt64 default_max_bytes_in_join;
    extern const SettingsBool join_use_nulls;
}

namespace ErrorCodes
{
    extern const int NOT_IMPLEMENTED;
    extern const int LOGICAL_ERROR;
    extern const int INVALID_JOIN_ON_EXPRESSION;
    extern const int NOT_FOUND_COLUMN_IN_BLOCK;
    extern const int INCORRECT_DATA;
}

std::optional<ASOFJoinInequality> operatorToAsofInequality(JoinConditionOperator op)
{
    switch (op)
    {
        case JoinConditionOperator::Less: return ASOFJoinInequality::Less;
        case JoinConditionOperator::LessOrEquals: return ASOFJoinInequality::LessOrEquals;
        case JoinConditionOperator::Greater: return ASOFJoinInequality::Greater;
        case JoinConditionOperator::GreaterOrEquals: return ASOFJoinInequality::GreaterOrEquals;
        default: return {};
    }
}

JoinStepLogical::JoinStepLogical(
    const Block & left_header_,
    const Block & right_header_,
    JoinInfo join_info_,
    JoinExpressionActions join_expression_actions_,
    Names required_output_columns_,
    bool use_nulls_,
    JoinSettings join_settings_,
    SortingStep::Settings sorting_settings_)
    : expression_actions(std::move(join_expression_actions_))
    , join_info(std::move(join_info_))
    , required_output_columns(std::move(required_output_columns_))
    , use_nulls(use_nulls_)
    , join_settings(std::move(join_settings_))
    , sorting_settings(std::move(sorting_settings_))
{
    updateInputHeaders({left_header_, right_header_});
}

QueryPipelineBuilderPtr JoinStepLogical::updatePipeline(QueryPipelineBuilders /* pipelines */, const BuildQueryPipelineSettings & /* settings */)
{
    throw Exception(ErrorCodes::NOT_IMPLEMENTED, "Cannot execute JoinStepLogical, it should be converted physical step first");
}

void JoinStepLogical::describePipeline(FormatSettings & settings) const
{
    IQueryPlanStep::describePipeline(processors, settings);
}

String formatJoinCondition(const std::vector<JoinActionRef> & predicates)
{
    return fmt::format("{}", fmt::join(predicates | std::views::transform([](const auto & x) { return x.getColumnName(); }), " AND "));
}

std::vector<std::pair<String, String>> JoinStepLogical::describeJoinActions() const
{
    std::vector<std::pair<String, String>> description;

    description.emplace_back("Type", toString(join_info.kind));
    description.emplace_back("Strictness", toString(join_info.strictness));
    description.emplace_back("Locality", toString(join_info.locality));
    description.emplace_back("Expression", formatJoinCondition(join_info.expression));

    description.emplace_back("Required Output", fmt::format("[{}]", fmt::join(required_output_columns, ", ")));

    for (const auto & [name, value] : runtime_info_description)
        description.emplace_back(name, value);

    return description;
}

void JoinStepLogical::describeActions(FormatSettings & settings) const
{
    String prefix(settings.offset, settings.indent_char);

    for (const auto & [name, value] : describeJoinActions())
        settings.out << prefix << name << ": " << value << '\n';

    settings.out << prefix << "Expression:\n";
    auto actions_dag = expression_actions.getActionsDAG()->clone();
    ExpressionActions(std::move(actions_dag)).describeActions(settings.out, prefix);
}

void JoinStepLogical::describeActions(JSONBuilder::JSONMap & map) const
{
    for (const auto & [name, value] : describeJoinActions())
        map.add(name, value);

    auto actions_dag = expression_actions.getActionsDAG()->clone();
    map.add("Actions", ExpressionActions(std::move(actions_dag)).toTree());
}

std::vector<JoinActionRef> getOutputColumns(
    const JoinExpressionActions & expression_actions,
    const Names & required_output_columns,
    bool use_nulls,
    const JoinInfo & join_info)
{
    std::vector<JoinActionRef> output_columns;
    for (const auto & out_column : required_output_columns)
    {
        auto node = expression_actions.findNode(out_column, true);
        if (use_nulls && isRightOrFull(join_info.kind) && node.getExpressionSources() == JoinTableSide::Left)
            output_columns.push_back(expression_actions.addCastToNullable(node));
        else if (use_nulls && isLeftOrFull(join_info.kind) && node.getExpressionSources() == JoinTableSide::Right)
            output_columns.push_back(expression_actions.addCastToNullable(node));
        else
            output_columns.push_back(node);
    }
    return output_columns;
}

void JoinStepLogical::updateOutputHeader()
{
    Header & header = output_header.emplace();
    auto output_columns = getOutputColumns(expression_actions, required_output_columns, use_nulls, join_info);
    for (auto out_node : output_columns)
        header.insert(out_node.getColumn());
}

/// When we have an expression like `a AND b`, it can work even when `a` and `b` are non-boolean values,
/// because the AND operator will implicitly convert them to booleans. The result will be either boolean or nullable.
/// In some cases we need to split `a` and `b` into separate expressions, but we want to preserve the same
/// boolean conversion behavior as if they were still part of the original AND expression.
JoinActionRef toBoolIfNeeded(JoinActionRef condition)
{
    auto output_type = removeNullable(condition.getType());
    WhichDataType which_type(output_type);
    if (!which_type.isUInt8())
        return JoinExpressionActions::addCast(condition, std::make_shared<DataTypeUInt8>());
    return condition;
}

auto nodeSourceFilter(BaseRelsSet filter_, bool strict = false)
{
    return [filter = std::move(filter_), strict](const auto & node)
    {
        if (strict)
            return node.getExpressionSources() == filter;
        return isSubsetOf(node.getExpressionSources(), filter);
    };
}

bool canPushDownFromOn(const JoinInfo & join_info, std::optional<JoinTableSide> side = {})
{
    bool is_suitable_kind = join_info.kind == JoinKind::Inner
        || join_info.kind == JoinKind::Cross
        || join_info.kind == JoinKind::Comma
        || join_info.kind == JoinKind::Paste
        || (side == JoinTableSide::Left && join_info.kind == JoinKind::Right)
        || (side == JoinTableSide::Right && join_info.kind == JoinKind::Left);

    return is_suitable_kind && join_info.strictness == JoinStrictness::All;
}

void predicateOperandsToCommonType(JoinActionRef & left_node, JoinActionRef & right_node)
{
    const auto & left_type = left_node.getType();
    const auto & right_type = right_node.getType();

    if (left_type->equals(*right_type))
        return;

    DataTypePtr common_type;
    try
    {
        common_type = getLeastSupertype(DataTypes{left_type, right_type});
    }
    catch (Exception & ex)
    {
        ex.addMessage("JOIN cannot infer common type in ON section for keys. Left key '{}' type {}. Right key '{}' type {}",
            left_node.getColumnName(), left_type->getName(),
            right_node.getColumnName(), right_type->getName());
        throw;
    }

    if (!left_type->equals(*common_type))
        left_node = JoinExpressionActions::addCast(left_node, common_type);

    if (!right_type->equals(*common_type))
        right_node = JoinExpressionActions::addCast(right_node, common_type);
}

std::tuple<JoinConditionOperator, JoinActionRef, JoinActionRef> asJoinOperator(JoinActionRef predicate, bool & reversed)
{
    JoinActionRef lhs(nullptr);
    JoinActionRef rhs(nullptr);
    auto predicate_op = predicate.asFunction({lhs, rhs});
    if (predicate_op == JoinConditionOperator::Unknown)
        return {JoinConditionOperator::Unknown, lhs, rhs};

    reversed = false;
    if (lhs.getExpressionSources() == JoinTableSide::Left && rhs.getExpressionSources() == JoinTableSide::Right)
    {
        return {predicate_op, lhs, rhs};
    }

    if (lhs.getExpressionSources() == JoinTableSide::Right && rhs.getExpressionSources() == JoinTableSide::Left)
    {
        reversed = true;
        return {predicate_op, rhs, lhs};
    }

    return {JoinConditionOperator::Unknown, nullptr, nullptr};
}

std::tuple<JoinConditionOperator, JoinActionRef, JoinActionRef> asJoinOperator(JoinActionRef predicate)
{
    bool reversed = false;
    return asJoinOperator(predicate, reversed);
}

bool addJoinPredicatesToTableJoin(std::vector<JoinActionRef> & predicates, TableJoin::JoinOnClause & table_join_clause, std::unordered_set<JoinActionRef> & used_expressions)
{
    bool has_join_predicates = false;
    std::vector<JoinActionRef> new_predicates;
    for (size_t i = 0; i < predicates.size(); ++i)
    {
        auto & predicate = new_predicates.emplace_back(std::move(predicates[i]));
        auto [predicate_op, lhs, rhs] = asJoinOperator(predicate);
        if (predicate_op != JoinConditionOperator::Equals && predicate_op != JoinConditionOperator::NullSafeEquals)
            continue;

        predicateOperandsToCommonType(lhs, rhs);
        bool null_safe_comparison = JoinConditionOperator::NullSafeEquals == predicate_op;
        if (null_safe_comparison && isNullableOrLowCardinalityNullable(lhs.getType()) && isNullableOrLowCardinalityNullable(rhs.getType()))
        {
            /**
                * In case of null-safe comparison (a IS NOT DISTINCT FROM b),
                * we need to wrap keys with a non-nullable type.
                * The type `tuple` can be used for this purpose,
                * because value tuple(NULL) is not NULL itself (moreover it has type Tuple(Nullable(T) which is not Nullable).
                * Thus, join algorithm will match keys with values tuple(NULL).
                * Example:
                *   SELECT * FROM t1 JOIN t2 ON t1.a <=> t2.b
                * This will be semantically transformed to:
                *   SELECT * FROM t1 JOIN t2 ON tuple(t1.a) == tuple(t2.b)
                */

            FunctionOverloadResolverPtr wrap_nullsafe_function = std::make_shared<FunctionToOverloadResolverAdaptor>(std::make_shared<FunctionTuple>());

            lhs = JoinExpressionActions::addFunction({lhs}, wrap_nullsafe_function);
            rhs = JoinExpressionActions::addFunction({rhs}, wrap_nullsafe_function);
        }

        has_join_predicates = true;
        table_join_clause.addKey(lhs.getColumnName(), rhs.getColumnName(), null_safe_comparison);

        /// We applied predicate, do not add it to residual conditions
        used_expressions.insert(lhs);
        used_expressions.insert(rhs);
        new_predicates.pop_back();
    }

    predicates = std::move(new_predicates);

    return has_join_predicates;
}


void JoinStepLogical::setPreparedJoinStorage(PreparedJoinStorage storage) { prepared_join_storage = std::move(storage); }

static Block blockWithColumns(ColumnsWithTypeAndName columns)
{
    Block block;
    for (const auto & column : columns)
        block.insert(ColumnWithTypeAndName(column.column ? column.column : column.type->createColumn(), column.type, column.name));
    return block;
}

using QueryPlanNode = QueryPlan::Node;
using QueryPlanNodePtr = QueryPlanNode *;

JoinActionRef concatConditions(std::vector<JoinActionRef> & conditions, BaseRelsSet source_filter)
{
    auto matching_point = std::ranges::partition(conditions, nodeSourceFilter(std::move(source_filter)));

    std::vector<JoinActionRef> matching(conditions.begin(), matching_point.begin());
    JoinActionRef result(nullptr);
    if (matching.size() == 1)
        result = toBoolIfNeeded(matching.front());
    else if (matching.size() > 1)
        result = JoinExpressionActions::addFunction(matching, JoinConditionOperator::And);

    conditions.erase(conditions.begin(), matching_point.begin());
    return result;
}

bool tryAddDisjunctiveConditions(
    std::vector<JoinActionRef> & join_expressions,
    TableJoin::Clauses & table_join_clauses,
    std::unordered_set<JoinActionRef> & used_expressions,
    bool throw_on_error = false)
{
    if (join_expressions.size() != 1)
        return false;

    auto & join_expression = join_expressions.front();
    if (join_expression.getFunction() != JoinConditionOperator::Or)
        return false;

    std::vector<JoinActionRef> disjunctive_conditions = join_expression.getArguments();
    bool has_residual_condition = false;
    for (auto expr : disjunctive_conditions)
    {
        std::vector<JoinActionRef> join_condition = {expr};
        if (expr.getFunction() == JoinConditionOperator::And)
            join_condition = expr.getArguments();

        auto & table_join_clause = table_join_clauses.emplace_back();
        bool has_keys = addJoinPredicatesToTableJoin(join_condition, table_join_clause, used_expressions);
        if (!has_keys && throw_on_error)
            throw Exception(ErrorCodes::INVALID_JOIN_ON_EXPRESSION, "Cannot determine join keys in JOIN ON expression {}",
                formatJoinCondition({expr}));
        if (!has_keys && !throw_on_error)
            return false;

        if (auto left_pre_filter_condition = concatConditions(join_condition, BaseRelsSet(8, 1)))
        {
            table_join_clause.analyzer_left_filter_condition_column_name = left_pre_filter_condition.getColumnName();
            used_expressions.insert(left_pre_filter_condition);
        }

        if (auto right_pre_filter_condition = concatConditions(join_condition, BaseRelsSet(8, 2)))
        {
            table_join_clause.analyzer_right_filter_condition_column_name = right_pre_filter_condition.getColumnName();
            used_expressions.insert(right_pre_filter_condition);
        }

        if (!join_condition.empty())
            has_residual_condition = true;
    }

    /// Clear join_expressions if there is no unhandled conditions, no need to calculate residual filter
    if (!has_residual_condition)
        join_expressions.clear();

    return true;
}

static void addSortingForMergeJoin(
    const FullSortingMergeJoin * join_ptr,
    QueryPlan::Node *& left_node,
    QueryPlan::Node *& right_node,
    QueryPlan::Nodes & nodes,
    const SortingStep::Settings & sort_settings,
    const JoinSettings & join_settings,
    const TableJoin & table_join)
{
    auto join_kind = table_join.kind();
    auto join_strictness = table_join.strictness();
    auto add_sorting = [&] (QueryPlan::Node *& node, const Names & key_names, JoinTableSide join_table_side)
    {
        SortDescription sort_description;
        sort_description.reserve(key_names.size());
        for (const auto & key_name : key_names)
            sort_description.emplace_back(key_name);

        auto sorting_step = std::make_unique<SortingStep>(
            node->step->getOutputHeader(), std::move(sort_description), 0 /*limit*/, sort_settings, true /*is_sorting_for_merge_join*/);
        sorting_step->setStepDescription(fmt::format("Sort {} before JOIN", join_table_side));
        node = &nodes.emplace_back(QueryPlan::Node{std::move(sorting_step), {node}});
    };

    auto crosswise_connection = CreateSetAndFilterOnTheFlyStep::createCrossConnection();
    auto add_create_set = [&](QueryPlan::Node *& node, const Names & key_names, JoinTableSide join_table_side)
    {
        auto creating_set_step = std::make_unique<CreateSetAndFilterOnTheFlyStep>(
            node->step->getOutputHeader(), key_names, join_settings.max_rows_in_set_to_optimize_join, crosswise_connection, join_table_side);
        creating_set_step->setStepDescription(fmt::format("Create set and filter {} joined stream", join_table_side));

        auto * step_raw_ptr = creating_set_step.get();
        node = &nodes.emplace_back(QueryPlan::Node{std::move(creating_set_step), {node}});
        return step_raw_ptr;
    };

    const auto & join_clause = join_ptr->getTableJoin().getOnlyClause();

    bool join_type_allows_filtering = (join_strictness == JoinStrictness::All || join_strictness == JoinStrictness::Any)
                                    && (isInner(join_kind) || isLeft(join_kind) || isRight(join_kind));

    auto has_non_const = [](const Block & block, const auto & keys)
    {
        for (const auto & key : keys)
        {
            const auto & column = block.getByName(key).column;
            if (column && !isColumnConst(*column))
                return true;
        }
        return false;
    };

    /// This optimization relies on the sorting that should buffer data from both streams before emitting any rows.
    /// Sorting on a stream with const keys can start returning rows immediately and pipeline may stuck.
    /// Note: it's also doesn't work with the read-in-order optimization.
    /// No checks here because read in order is not applied if we have `CreateSetAndFilterOnTheFlyStep` in the pipeline between the reading and sorting steps.
    bool has_non_const_keys = has_non_const(left_node->step->getOutputHeader(), join_clause.key_names_left)
        && has_non_const(right_node->step->getOutputHeader() , join_clause.key_names_right);
    if (join_settings.max_rows_in_set_to_optimize_join > 0 && join_type_allows_filtering && has_non_const_keys)
    {
        auto * left_set = add_create_set(left_node, join_clause.key_names_left, JoinTableSide::Left);
        auto * right_set = add_create_set(right_node, join_clause.key_names_right, JoinTableSide::Right);

        if (isInnerOrLeft(join_kind))
            right_set->setFiltering(left_set->getSet());

        if (isInnerOrRight(join_kind))
            left_set->setFiltering(right_set->getSet());
    }

    add_sorting(left_node, join_clause.key_names_left, JoinTableSide::Left);
    add_sorting(right_node, join_clause.key_names_right, JoinTableSide::Right);
}


static void constructPhysicalStep(
    QueryPlanNode & node,
    ActionsDAG left_pre_join_actions,
    ActionsDAG post_join_actions,
    String residual_filter_condition,
    JoinPtr join_ptr,
    const QueryPlanOptimizationSettings & ,
    const JoinSettings & join_settings,
    const SortingStep::Settings &,
    QueryPlan::Nodes & nodes)
{
    if (node.children.size() != 1)
        throw Exception(ErrorCodes::LOGICAL_ERROR, "Expected 1 child, got {}", node.children.size());

    if (!join_ptr->isFilled())
        throw Exception(ErrorCodes::LOGICAL_ERROR, "Join is not filled");

    auto * join_left_node = node.children[0];
    makeExpressionNodeOnTopOf(*join_left_node, std::move(left_pre_join_actions), {}, nodes);

    node.step = std::make_unique<FilledJoinStep>(
        join_left_node->step->getOutputHeader(),
        join_ptr,
        join_settings.max_block_size);
    makeExpressionNodeOnTopOf(node, std::move(post_join_actions), residual_filter_condition, nodes);
}

static void constructPhysicalStep(
    QueryPlanNode & node,
    ActionsDAG left_pre_join_actions,
    ActionsDAG right_pre_join_actions,
    ActionsDAG post_join_actions,
    String residual_filter_condition,
    JoinPtr join_ptr,
    const QueryPlanOptimizationSettings & optimization_settings,
    const JoinSettings & join_settings,
    const SortingStep::Settings & sorting_settings,
    QueryPlan::Nodes & nodes)
{
    if (node.children.size() != 2)
        throw Exception(ErrorCodes::LOGICAL_ERROR, "Expected 2 children, got {}", node.children.size());

    if (join_ptr->isFilled())
        throw Exception(ErrorCodes::LOGICAL_ERROR, "Join is already filled");

    auto * join_left_node = node.children[0];
    auto * join_right_node = node.children[1];

    makeExpressionNodeOnTopOf(*join_left_node, std::move(left_pre_join_actions), {}, nodes);
    makeExpressionNodeOnTopOf(*join_right_node, std::move(right_pre_join_actions), {}, nodes);

    if (const auto * fsmjoin = dynamic_cast<const FullSortingMergeJoin *>(join_ptr.get()))
        addSortingForMergeJoin(fsmjoin, join_left_node, join_right_node, nodes,
            sorting_settings, join_settings, fsmjoin->getTableJoin());

    auto required_output_from_join = post_join_actions.getRequiredColumnsNames();
    node.step = std::make_unique<JoinStep>(
        join_left_node->step->getOutputHeader(),
        join_right_node->step->getOutputHeader(),
        join_ptr,
        join_settings.max_block_size,
        join_settings.min_joined_block_size_bytes,
        optimization_settings.max_threads,
        NameSet(required_output_from_join.begin(), required_output_from_join.end()),
        false /*optimize_read_in_order*/,
        true /*use_new_analyzer*/);
    makeExpressionNodeOnTopOf(node, std::move(post_join_actions), residual_filter_condition, nodes);
}

static QueryPlanNode buildPhysicalJoinImpl(
    std::vector<QueryPlanNode *> children,
    JoinInfo join_info,
    JoinExpressionActions expression_actions,
    JoinSettings join_settings,
    JoinAlgorithmParams join_algorithm_params,
    SortingStep::Settings sorting_settings,
    const Names & required_output_columns,
    bool use_nulls,
    PreparedJoinStorage prepared_join_storage,
    const QueryPlanOptimizationSettings & optimization_settings,
    QueryPlan::Nodes & nodes)
{
    auto table_join = std::make_shared<TableJoin>(join_settings, use_nulls,
        Context::getGlobalContextInstance()->getGlobalTemporaryVolume(),
        Context::getGlobalContextInstance()->getTempDataOnDisk());

    if (prepared_join_storage)
    {
        prepared_join_storage.visit([&table_join](const auto & storage_)
        {
            table_join->setStorageJoin(storage_);
        });
    }

    auto & join_expression = join_info.expression;

    std::unordered_set<JoinActionRef> used_expressions;

    bool is_disjunctive_condition = false;
    auto & table_join_clauses = table_join->getClauses();
    if (!isCrossOrComma(join_info.kind) && !isPaste(join_info.kind))
    {
        bool has_keys = addJoinPredicatesToTableJoin(join_expression, table_join_clauses.emplace_back(), used_expressions);

        if (!has_keys)
        {
            bool can_convert_to_cross = (isInner(join_info.kind) || isCrossOrComma(join_info.kind))
                && TableJoin::isEnabledAlgorithm(join_settings.join_algorithms, JoinAlgorithm::HASH)
                && join_info.strictness == JoinStrictness::All;

            is_disjunctive_condition = tryAddDisjunctiveConditions(
                join_expression, table_join_clauses, used_expressions, !can_convert_to_cross);

            if (!is_disjunctive_condition && !can_convert_to_cross)
                throw Exception(ErrorCodes::INVALID_JOIN_ON_EXPRESSION, "Cannot determine join keys in JOIN ON expression {}",
                    formatJoinCondition(join_expression));
            join_info.kind = JoinKind::Cross;
            table_join_clauses.pop_back();
        }
    }
    else if (!join_expression.empty())
    {
        throw Exception(ErrorCodes::INVALID_JOIN_ON_EXPRESSION, "Unexpected JOIN ON expression {} for {} JOIN",
            formatJoinCondition(join_expression), toString(join_info.kind));
    }

    if (join_info.strictness == JoinStrictness::Asof)
    {
        if (is_disjunctive_condition)
            throw Exception(ErrorCodes::INVALID_JOIN_ON_EXPRESSION, "ASOF join does not support multiple disjuncts in JOIN ON expression");

        /// Find strictly only one inequality in predicate list for ASOF join
        chassert(table_join_clauses.size() == 1);
        auto found_asof_predicate_it = join_expression.end();
        for (auto it = join_expression.begin(); it != join_expression.end(); ++it)
        {
            bool reversed = false;
            auto [predicate_op, lhs, rhs] = asJoinOperator(*it, reversed);
            auto asof_inequality_op = operatorToAsofInequality(predicate_op);
            if (!asof_inequality_op)
                continue;
            if (reversed)
                *asof_inequality_op = reverseASOFJoinInequality(*asof_inequality_op);

            if (found_asof_predicate_it != join_expression.end())
                throw Exception(ErrorCodes::INVALID_JOIN_ON_EXPRESSION, "ASOF join does not support multiple inequality predicates in JOIN ON expression");
            found_asof_predicate_it = it;

            predicateOperandsToCommonType(lhs, rhs);

            used_expressions.insert(lhs);
            used_expressions.insert(rhs);

            table_join->setAsofInequality(*asof_inequality_op);
            table_join_clauses.front().addKey(lhs.getColumnName(), rhs.getColumnName(), /* null_safe_comparison = */ false);
        }
        if (found_asof_predicate_it == join_expression.end())
            throw Exception(ErrorCodes::INVALID_JOIN_ON_EXPRESSION, "ASOF join requires one inequality predicate in JOIN ON expression, in {}",
                formatJoinCondition(join_expression));

        join_expression.erase(found_asof_predicate_it);
    }

    if (auto left_pre_filter_condition = concatConditions(join_expression, BaseRelsSet(8, 1)))
    {
        table_join_clauses.at(table_join_clauses.size() - 1).analyzer_left_filter_condition_column_name = left_pre_filter_condition.getColumnName();
        used_expressions.insert(left_pre_filter_condition);
    }

    if (auto right_pre_filter_condition = concatConditions(join_expression, BaseRelsSet(8, 2)))
    {
        table_join_clauses.at(table_join_clauses.size() - 1).analyzer_right_filter_condition_column_name = right_pre_filter_condition.getColumnName();
        used_expressions.insert(right_pre_filter_condition);
    }

    JoinActionRef residual_filter_condition = concatConditions(join_expression, BaseRelsSet(8, 3));

    if (!join_expression.empty())
        throw Exception(ErrorCodes::LOGICAL_ERROR, "Unhandled join conditions: {}", formatJoinCondition(join_expression));

    auto output_columns = getOutputColumns(expression_actions, required_output_columns, use_nulls, join_info);
    for (const auto & out_column : output_columns)
        used_expressions.insert(out_column);

    if (residual_filter_condition && !is_disjunctive_condition && canPushDownFromOn(join_info))
    {
        auto residual_inputs = residual_filter_condition.getArguments(true);
        for (const auto & input : residual_inputs)
        {
            auto [_, inserted] = used_expressions.emplace(input);
            if (inserted)
                output_columns.push_back(input);
        }
    }
    else if (residual_filter_condition)
    {
        auto residual_filter_dag = expression_actions.getSubDAG(std::views::single(residual_filter_condition));
        ExpressionActionsPtr & mixed_join_expression = table_join->getMixedJoinExpression();
        mixed_join_expression = std::make_shared<ExpressionActions>(std::move(residual_filter_dag), optimization_settings.actions_settings);
        residual_filter_condition = JoinActionRef(nullptr);
    }

    ActionsDAG left_dag = expression_actions.getSubDAG(used_expressions | std::views::filter(nodeSourceFilter(BaseRelsSet(8, 1), false)));
    ActionsDAG right_dag = expression_actions.getSubDAG(used_expressions | std::views::filter(nodeSourceFilter(BaseRelsSet(8, 2), true)));

    ActionsDAG residual_dag;
    for (const auto & column : output_columns)
        residual_dag.addInput(column.getColumnName(), column.getType());

    if (residual_filter_condition)
    {
        auto residual_filter_dag = expression_actions.getSubDAG(std::views::single(residual_filter_condition));
        residual_dag.mergeInplace(std::move(residual_filter_dag));
    }

    const auto & residual_inputs = residual_dag.getInputs();
    auto & residual_outputs = residual_dag.getOutputs();
    for (const auto [input, action_node] : std::views::zip(residual_inputs, output_columns))
    {
        if (input->result_name == action_node.getColumnName())
            residual_outputs.push_back(input);
        else
            residual_outputs.push_back(&residual_dag.addAlias(*input, action_node.getColumnName()));
    }

    table_join->setInputColumns(
        left_dag.getNamesAndTypesList(),
        right_dag.getNamesAndTypesList());
    table_join->setUsedColumns(residual_dag.getRequiredColumnsNames());
    table_join->setJoinInfo(join_info);

    Block left_sample_block = blockWithColumns(left_dag.getResultColumns());
    Block right_sample_block = blockWithColumns(right_dag.getResultColumns());

    auto join_algorithm_ptr = chooseJoinAlgorithm(
        table_join,
        prepared_join_storage,
        left_sample_block,
        right_sample_block,
        join_algorithm_params);

    QueryPlanNode node;
    node.children = std::move(children);
    String residual_filter_condition_name = residual_filter_condition ? residual_filter_condition.getColumnName() : "";
    if (!join_algorithm_ptr->isFilled())
    {
        constructPhysicalStep(
            node, std::move(left_dag), std::move(right_dag), std::move(residual_dag), residual_filter_condition_name,
            std::move(join_algorithm_ptr), optimization_settings, join_settings, sorting_settings, nodes);
    }
    else
    {
        if (!right_dag.trivial())
            throw Exception(ErrorCodes::LOGICAL_ERROR, "Expected right stream to be trivial, got {}", right_dag.dumpDAG());
        constructPhysicalStep(
            node, std::move(left_dag), std::move(residual_dag), residual_filter_condition_name,
            std::move(join_algorithm_ptr), optimization_settings, join_settings, sorting_settings, nodes);
    }
    return node;
}

void JoinStepLogical::buildPhysicalJoin(
    QueryPlanNode & node,
    std::vector<RelationStats> relation_stats,
    const QueryPlanOptimizationSettings & optimization_settings,
    QueryPlan::Nodes & nodes)
{
    auto * join_step = typeid_cast<JoinStepLogical *>(node.step.get());
    if (!join_step)
    {
        if (node.step)
        {
            const auto & step = *node.step;
            auto type_name = demangle(typeid(step).name());
            throw Exception(ErrorCodes::LOGICAL_ERROR, "Expected JoinStepLogical, got '{}' of type {}", node.step->getName(), type_name);
        }
        throw Exception(ErrorCodes::LOGICAL_ERROR, "Expected JoinStepLogical, got nullptr");
    }

    JoinAlgorithmParams join_algorithm_params(
        join_step->join_settings,
        optimization_settings.max_threads,
        join_step->hash_table_key_hashes->key_hash_left,
        optimization_settings.max_entries_for_hash_table_stats,
        optimization_settings.initial_query_id,
        optimization_settings.lock_acquire_timeout);

    if (relation_stats.size() == 2 && relation_stats[1].estimated_rows > 0)
        join_algorithm_params.rhs_size_estimation = relation_stats[1].estimated_rows;

    auto new_node = buildPhysicalJoinImpl(
        node.children,
        join_step->join_info,
        std::move(join_step->expression_actions),
        join_step->join_settings,
        join_algorithm_params,
        join_step->sorting_settings,
        join_step->required_output_columns,
        join_step->use_nulls,
        join_step->prepared_join_storage,
        optimization_settings,
        nodes);

    node = std::move(new_node);
}

bool JoinStepLogical::hasPreparedJoinStorage() const
{
    return prepared_join_storage;
}

std::optional<ActionsDAG> JoinStepLogical::getFilterActions(JoinTableSide side, String & filter_column_name)
{
    if (join_info.strictness != JoinStrictness::All)
        return {};

    auto & join_expression = join_info.expression;

    if (!canPushDownFromOn(join_info, side))
        return {};

    if (auto filter_condition = concatConditions(join_expression, BaseRelsSet(side)))
    {
        filter_column_name = filter_condition.getColumnName();
        ActionsDAG new_dag = JoinExpressionActions::getSubDAG(filter_condition);
        if (new_dag.getOutputs().size() != 1)
            throw Exception(ErrorCodes::LOGICAL_ERROR, "Expected 1 output column, got {}", new_dag.getOutputs().size());

        const auto & inputs = new_dag.getInputs();
        auto & outputs = new_dag.getOutputs();
        if (std::ranges::contains(inputs, outputs.front()))
            outputs.clear();
        outputs.append_range(inputs);

        return std::move(new_dag);
    }

    return {};
}

void JoinStepLogical::serializeSettings(QueryPlanSerializationSettings & settings) const
{
    join_settings.updatePlanSettings(settings);
    sorting_settings.updatePlanSettings(settings);
}

void JoinStepLogical::serialize(Serialization & ctx) const
{
    if (prepared_join_storage)
        throw Exception(ErrorCodes::NOT_IMPLEMENTED, "Serialization of JoinStepLogical with prepared storage is not implemented");

    UInt8 flags = 0;
    if (use_nulls)
        flags |= 1;

    writeIntBinary(flags, ctx.out);

    writeVarUInt(1, ctx.out);
    auto actions_dag = expression_actions.getActionsDAG();
    actions_dag->serialize(ctx.out, ctx.registry);

    join_info.serialize(ctx.out, actions_dag.get());

    writeVarUInt(required_output_columns.size(), ctx.out);
    for (const auto & name : required_output_columns)
        writeStringBinary(name, ctx.out);
}

std::unique_ptr<IQueryPlanStep> JoinStepLogical::deserialize(Deserialization & ctx)
{
    if (ctx.input_headers.size() != 2)
        throw Exception(ErrorCodes::INCORRECT_DATA, "JoinStepLogical must have two input streams");

    UInt8 flags;
    readIntBinary(flags, ctx.in);

    bool use_nulls = flags & 1;

    ActionsDAG actions_dag;
    {
        UInt64 num_dags;
        readVarUInt(num_dags, ctx.in);

        if (num_dags != 1)
            throw Exception(ErrorCodes::INCORRECT_DATA, "JoinStepLogical deserialization expect 3 DAGs, got {}", num_dags);

        actions_dag = ActionsDAG::deserialize(ctx.in, ctx.registry, ctx.context);
    }

    auto left_header = ctx.input_headers.front();
    auto right_header = ctx.input_headers.back();
    JoinExpressionActions expression_actions(left_header, right_header, std::move(actions_dag));

    auto join_info = JoinInfo::deserialize(ctx.in, expression_actions);

    Names required_output_columns;
    {
        UInt64 num_output_columns;
        readVarUInt(num_output_columns, ctx.in);

        required_output_columns.resize(num_output_columns);
        for (auto & name : required_output_columns)
            readStringBinary(name, ctx.in);
    }

    SortingStep::Settings sort_settings(ctx.settings);
    JoinSettings join_settings(ctx.settings);

    return std::make_unique<JoinStepLogical>(
        std::move(left_header),
        std::move(right_header),
        std::move(join_info),
        std::move(expression_actions),
        std::move(required_output_columns),
        use_nulls,
        std::move(join_settings),
        std::move(sort_settings));
}

QueryPlanStepPtr JoinStepLogical::clone() const
{
    auto new_join_info = join_info;
    auto new_expression_actions = expression_actions.clone(new_join_info.expression);

    auto result_step = std::make_unique<JoinStepLogical>(
        getInputHeaders().front(), getInputHeaders().back(),
        std::move(new_join_info),
        std::move(new_expression_actions),
        required_output_columns,
        use_nulls,
        join_settings,
        sorting_settings);
    result_step->setStepDescription(getStepDescription());
    return result_step;
}

void JoinStepLogical::addConditions(ActionsDAG::NodeRawConstPtrs condition_nodes)
{
    auto conditions = expression_actions.addNodes(condition_nodes);
    join_info.expression.append_range(conditions);
}

void registerJoinStep(QueryPlanStepRegistry & registry)
{
    registry.registerStep("Join", JoinStepLogical::deserialize);
}


}
