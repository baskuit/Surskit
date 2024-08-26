#pragma once

#include <libpinyon/math.h>
#include <state/state.h>
#include <tree/node.h>

template <IsStateTypes Types, typename MStats, typename CStats,
          typename NodeActions = void, typename NodeValue = void>
struct LNodes : Types
{
    /*
    Instead of MatrixNodes storing their obs memeber, they are bundled with 'Edge' struct
    which is basically a linked list that the chance node parent owns
    */

    friend std::ostream &operator<<(std::ostream &os, const LNodes &)
    {
        os << "LNodes";
        return os;
    }

    class MatrixNode;

    class ChanceNode;

    using MatrixStats = MStats;
    using ChanceStats = CStats;

    class MatrixNode : public MatrixNodeData<Types, NodeActions, NodeValue>
    {
    public:
        ChanceNode *child = nullptr;

        bool terminal = false;
        bool expanded = false;

        MatrixStats stats;

        MatrixNode(){};
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
        struct Edge
        {
            MatrixNode *matrix_node = nullptr;
            typename Types::Obs obs;
            Edge *next = nullptr;
            Edge() {}
            Edge(
                MatrixNode *matrix_node,
                typename Types::Obs obs) : matrix_node(matrix_node), obs(obs) {}
            ~Edge()
            {
                delete matrix_node;
                delete next;
            }
        };

        ChanceNode *next = nullptr;
        Edge *edge = nullptr;

        int row_idx;
        int col_idx;

        ChanceStats stats;

        ChanceNode() {}
        ChanceNode(
            int row_idx,
            int col_idx) : row_idx(row_idx), col_idx(col_idx) {}
        ChanceNode(const ChanceNode &) = delete;
        ~ChanceNode();

        MatrixNode *access(const Types::Obs &obs)
        {
            if (this->edge == nullptr)
            {
                this->edge = new Edge(new MatrixNode(), obs);
                return this->edge->matrix_node;
            }
            Edge *current = this->edge;
            Edge *previous = this->edge;
            while (current != nullptr)
            {
                previous = current;
                if (current->obs == obs)
                {
                    return current->matrix_node;
                }
                current = current->next;
            }
            Edge *new_edge = new Edge(new MatrixNode(), obs);
            previous->next = new_edge;
            return new_edge->matrix_node;
        };

        const MatrixNode *access(const Types::Obs &obs) const
        {
            if (this->edge == nullptr)
            {
                return nullptr;
            }
            const Edge *current = this->edge;
            const Edge *previous = this->edge;
            while (current != nullptr)
            {
                previous = current;
                if (current->obs == obs)
                {
                    return current->matrix_node;
                }
                current = current->next;
            }
            return nullptr;
        };

        size_t count_matrix_nodes()
        {
            size_t c = 0;
            auto current = this->edge;
            while (current != nullptr)
            {
                c += current->matrix_node->count_matrix_nodes();
                current = current->next;
            }
            return c;
        }
    };
};

template <IsStateTypes Types, typename MStats, typename CStats,
          typename stores_actions, typename stores_value>
LNodes<Types, MStats, CStats, stores_actions, stores_value>::MatrixNode::~MatrixNode()
{
    while (this->child != nullptr)
    {
        LNodes<Types, MStats, CStats, stores_actions, stores_value>::ChanceNode *victim = this->child;
        this->child = this->child->next;
        delete victim;
    }
}
template <IsStateTypes Types, typename MStats, typename CStats,
          typename stores_actions, typename stores_value>
LNodes<Types, MStats, CStats, stores_actions, stores_value>::ChanceNode::~ChanceNode()
{
    delete this->edge;
};