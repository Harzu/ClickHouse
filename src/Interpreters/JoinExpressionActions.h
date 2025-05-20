#pragma once

#include <Interpreters/ActionsDAG.h>
#include <bitset>
#include <initializer_list>
#include <memory>
#include <utility>
#include <ranges>
#include <vector>
#include <boost/dynamic_bitset.hpp>
#include <Core/Joins.h>

namespace DB
{

namespace ErrorCodes
{
    extern const int LOGICAL_ERROR;
}

enum class JoinConditionOperator : UInt8
{
    And,
    Or,
    Equals,
    NullSafeEquals,
    Less,
    LessOrEquals,
    Greater,
    GreaterOrEquals,
    Unknown,
};

std::string_view toString(JoinConditionOperator op);

class BaseRelsSet : public boost::dynamic_bitset<>
{
public:
    using Base = boost::dynamic_bitset<>;
    using Base::dynamic_bitset;

    explicit BaseRelsSet(JoinTableSide side) : Base(2) /// NOLINT
    {
        if (side == JoinTableSide::Left)
            Base::set(0);
        else if (side == JoinTableSide::Right)
            Base::set(1);
        else
            throw Exception(ErrorCodes::LOGICAL_ERROR, "Illegal value for JoinTableSide: {}", static_cast<Int32>(side));
    }
};

inline bool operator==(const BaseRelsSet & lhs, JoinTableSide side) { return lhs == BaseRelsSet(side); }
inline bool isSubsetOf(const BaseRelsSet & lhs, const BaseRelsSet & rhs) { return (lhs & rhs) == lhs; }

class JoinActionRef;

template <typename T>
class SafeSharedPtr
{
public:
    SafeSharedPtr(std::shared_ptr<T> ptr_) : ptr(ptr_) { assertNotNull(); } /// NOLINT
    operator bool() const { return ptr != nullptr; } /// NOLINT
    T & operator*() const { assertNotNull(); return *ptr; } /// NOLINT
    T * operator->() const { assertNotNull(); return ptr.get(); } /// NOLINT
    T * get() const { assertNotNull(); return ptr.get(); } /// NOLINT

    std::shared_ptr<T> stdPtr() const { return ptr; }

private:
    void assertNotNull() const
    {
        if (unlikely(!ptr))
            throw Exception(ErrorCodes::LOGICAL_ERROR, "Pointer is null");
    }

    std::shared_ptr<T> ptr;
};

class JoinExpressionActions
{
public:
    using NodeRawPtr = const ActionsDAG::Node *;

    JoinExpressionActions(const Block & left_header, const Block & right_header);
    JoinExpressionActions(const Block & left_header, const Block & right_header, ActionsDAG && actions_dag);

    static JoinActionRef addFunction(const std::vector<JoinActionRef> & arguments, const FunctionOverloadResolverPtr & function);
    static JoinActionRef addFunction(const std::vector<JoinActionRef> & arguments, JoinConditionOperator op);
    static JoinActionRef addCast(JoinActionRef action, DataTypePtr type);
    static JoinActionRef addCastToNullable(JoinActionRef action);

    JoinExpressionActions clone(std::vector<JoinActionRef> & nodes) const;

    std::vector<JoinActionRef> addNodes(ActionsDAG::NodeRawConstPtrs nodes);
    JoinActionRef findNode(const String & column_name, bool is_input = false) const;

    SafeSharedPtr<ActionsDAG> getActionsDAG() const;

    // template <std::ranges::input_range R>
    //     requires std::is_reference_v<std::ranges::range_reference_t<R>>
    // void foo(R&& range) {
    //     for (auto&& ref : range) {
    //         ref++;
    //     }
    // }

    template <std::ranges::range Range>
    requires std::convertible_to<std::ranges::range_value_t<Range>, JoinActionRef>
    static ActionsDAG getSubDAG(Range && range)
    {
        auto nodes = std::ranges::to<std::vector>(range | std::views::transform([](const auto & action) { return action.getNode(); }));
        return ActionsDAG::cloneSubDAG(nodes, /* remove_aliases= */ false);
    }

    static ActionsDAG getSubDAG(JoinActionRef action);


    JoinExpressionActions(const JoinExpressionActions &) = delete;
    JoinExpressionActions & operator=(const JoinExpressionActions &) = delete;

    JoinExpressionActions(JoinExpressionActions &&) = default;
    JoinExpressionActions & operator=(JoinExpressionActions &&) = default;

private:
    friend class JoinActionRef;

    struct Data;

    explicit JoinExpressionActions(std::shared_ptr<Data> data_) : data(data_) {}

    std::shared_ptr<Data> data;
};

class JoinActionRef
{
public:
    using NodeRawPtr = JoinExpressionActions::NodeRawPtr;

    JoinActionRef(std::nullptr_t) : actions_dag(nullptr) {} /// NOLINT

    explicit JoinActionRef(NodeRawPtr node_, JoinExpressionActions & expression_actions_);
    explicit JoinActionRef(NodeRawPtr node_, std::shared_ptr<JoinExpressionActions::Data> data_);

    NodeRawPtr getNode() const;

    ColumnWithTypeAndName getColumn() const;
    const String & getColumnName() const;
    DataTypePtr getType() const;

    operator bool() const { return actions_dag != nullptr; } /// NOLINT

    // void serialize(WriteBuffer & out) const;
    // static JoinActionRef deserialize(ReadBuffer & in, const ActionsDAG * actions_dag_);

    BaseRelsSet getExpressionSources() const;
    std::vector<JoinActionRef> getArguments(bool recursive = false) const;

    JoinConditionOperator getFunction() const;
    JoinConditionOperator asFunction(std::initializer_list<std::reference_wrapper<JoinActionRef>> operands) const;

private:
    std::shared_ptr<JoinExpressionActions::Data> getData() const;

    const ActionsDAG * actions_dag = nullptr;
    String column_name;
    std::weak_ptr<JoinExpressionActions::Data> data;
};

}

template <> struct std::hash<DB::JoinActionRef>
{
    size_t operator()(const DB::JoinActionRef & ref) const { return std::hash<const DB::ActionsDAG::Node *>()(ref.getNode()); }
};
