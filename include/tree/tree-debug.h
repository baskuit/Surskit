#pragma once

#include <libpinyon/math.h>
#include <state/state.h>
#include <tree/node.h>

template <IsStateTypes Types, typename MStats, typename CStats,
          typename NodeActions = void, typename NodeValue = void>
struct DebugNodes : Types
{

    friend std::ostream &operator<<(std::ostream &os, const DebugNodes &)
    {
        os << "DebugNodes";
        return os;
    }

    class MatrixNode;

    class ChanceNode;

    using MatrixStats = MStats;
    using ChanceStats = CStats;

    class MatrixNode : public MatrixNodeData<Types, NodeActions, NodeValue>
    {
    public:
        ChanceNode *parent = nullptr;
        ChanceNode *child = nullptr;
        MatrixNode *prev = nullptr;
        MatrixNode *next = nullptr;

        bool terminal = false;
        bool expanded = false;

        typename Types::Obs obs;
        MatrixStats stats;

        MatrixNode(){};
        MatrixNode(Types::Obs obs) : obs(obs) {}
        MatrixNode(
            ChanceNode *parent,
            MatrixNode *prev,
            Types::Obs obs) : parent(parent), prev(prev), obs(obs) {}
        MatrixNode(const MatrixNode &) = delete;
        ~MatrixNode();

        inline void expand(const size_t &, const size_t &)
        {
            expanded = true;
        }

        inline bool is_terminal() const
        {
            return terminal;
        }

        inline bool is_expanded() const
        {
            return expanded;
        }

        inline void set_terminal()
        {
            terminal = true;
        }

        inline void set_expanded()
        {
            expanded = true;
        }

        inline void get_value(Types::Value &value) const
        {
        }

        ChanceNode *access(int row_idx, int col_idx)
        {
            if (this->child == nullptr)
            {
                this->child = new ChanceNode(row_idx, col_idx);
                return this->child;
            }
            ChanceNode *current = this->child;
            ChanceNode *previous = this->child;
            while (current != nullptr)
            {
                previous = current;
                if (current->row_idx == row_idx && current->col_idx == col_idx)
                {
                    return current;
                }
                current = current->next;
            }
            ChanceNode *child = new ChanceNode(row_idx, col_idx);
            previous->next = child;
            return child;
        };

        const ChanceNode *access(int row_idx, int col_idx) const
        {
            if (this->child == nullptr)
            {
                return this->child;
            }
            const ChanceNode *current = this->child;
            const ChanceNode *previous = this->child;
            while (current != nullptr)
            {
                previous = current;
                if (current->row_idx == row_idx && current->col_idx == col_idx)
                {
                    return current;
                }
                current = current->next;
            }
            return current;
        };

        size_t count_siblings()
        {
            // called on a matrix node to see how many branches its chance node parent has
            size_t c = 1;
            MatrixNode *current = this->next;
            while (current != nullptr)
            {
                ++c;
                current = current->next;
            }
            current = this->prev;
            while (current != nullptr)
            {
                ++c;
                current = current->prev;
            }
            return c;
        }

        size_t count_matrix_nodes()
        {
            size_t c = 1;
            ChanceNode *current = this->child;
            while (current != nullptr)
            {
                c += current->count_matrix_nodes();
                current = current->next;
            }
            return c;
        }
    };

    class ChanceNode
    {
    public:
        MatrixNode *parent = nullptr;
        MatrixNode *child = nullptr;
        ChanceNode *prev = nullptr;
        ChanceNode *next = nullptr;

        int row_idx;
        int col_idx;

        ChanceStats stats;

        ChanceNode() {}
        ChanceNode(
            int row_idx,
            int col_idx) : row_idx(row_idx), col_idx(col_idx) {}
        ChanceNode(
            MatrixNode *parent,
            ChanceNode *prev,
            int row_idx,
            int col_idx) : parent(parent), prev(prev), row_idx(row_idx), col_idx(col_idx) {}
        ChanceNode(const ChanceNode &) = delete;
        ~ChanceNode();

        MatrixNode *access(const Types::Obs &obs)
        {
            if (this->child == nullptr)
            {
                MatrixNode *child = new MatrixNode(this, nullptr, obs);
                this->child = child;
                return child;
            }
            MatrixNode *current = this->child;
            MatrixNode *previous = this->child;
            while (current != nullptr)
            {
                previous = current;
                if (current->obs == obs)
                {
                    return current;
                }
                current = current->next;
            }
            MatrixNode *child = new MatrixNode(this, previous, obs);
            previous->next = child;
            return child;
        };

        const MatrixNode *access(const Types::Obs &obs) const
        {
            if (this->child == nullptr)
            {
                return this->child;
            }
            const MatrixNode *current = this->child;
            const MatrixNode *previous = this->child;
            while (current != nullptr)
            {
                previous = current;
                if (current->obs == obs)
                {
                    return current;
                }
                current = current->next;
            }
            return current;
        };

        size_t count_matrix_nodes()
        {
            size_t c = 0;
            MatrixNode *current = this->child;
            while (current != nullptr)
            {
                c += current->count_matrix_nodes();
                current = current->next;
            }
            return c;
        }
    };
};

template <IsStateTypes Types, typename MStats, typename CStats,
          typename stores_actions, typename stores_value>
DebugNodes<Types, MStats, CStats, stores_actions, stores_value>::MatrixNode::~MatrixNode()
{
    while (this->child != nullptr)
    {
        DebugNodes<Types, MStats, CStats, stores_actions, stores_value>::ChanceNode *victim = this->child;
        this->child = this->child->next;
        delete victim;
    }
}

template <IsStateTypes Types, typename MStats, typename CStats,
          typename stores_actions, typename stores_value>
DebugNodes<Types, MStats, CStats, stores_actions, stores_value>::ChanceNode::~ChanceNode()
{
    while (this->child != nullptr)
    {
        DebugNodes<Types, MStats, CStats, stores_actions, stores_value>::MatrixNode *victim = this->child;
        this->child = this->child->next;
        delete victim;
    }
};
