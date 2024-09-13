#pragma once

#include <libpinyon/math.h>
#include <state/state.h>
#include <tree/node.h>

#include <unordered_map>

template <IsStateTypes Types, typename MStats, typename CStats,
          typename NodeActions = void, typename NodeValue = void>
struct FlatNodes : Types
{

    friend std::ostream &operator<<(std::ostream &os, const FlatNodes &)
    {
        os << "FlatNodes";
        return os;
    }

    class MatrixNode;

    class ChanceNode;

    using MatrixStats = MStats;
    using ChanceStats = CStats;

    class MatrixNode : public MatrixNodeData<Types, NodeActions, NodeValue>
    {
    public:
        bool terminal = false;
        bool expanded = false;
        typename Types::Obs obs;
        int rows = 0;
        int cols = 0;

        MatrixStats stats;

        ChanceNode **edges;

        MatrixNode(){};
        MatrixNode(Types::Obs obs) : obs(obs) {}
        MatrixNode(const MatrixNode &) = delete;
        ~MatrixNode();

        inline void expand(const size_t &rows, const size_t &cols)
        {
            this->rows = rows;
            this->cols = cols;
            const size_t n_children = rows * cols;
            edges = new ChanceNode *[n_children] {};
            std::fill_n(edges, n_children, nullptr);
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
            const int child_idx = row_idx * cols + col_idx;
            ChanceNode *&child = edges[child_idx]; // ref to pointer
            if (child == nullptr)
            {
                child = new ChanceNode();
            }
            return child;
        };

        const ChanceNode *access(int row_idx, int col_idx) const
        {
            const int child_idx = row_idx * cols + col_idx;
            const ChanceNode *&child = edges[child_idx];
            return child;
        };

        ChanceNode *access(int row_idx, int col_idx, Types::Mutex &mutex)
        {
            const int child_idx = row_idx * cols + col_idx;
            const ChanceNode *&child = edges[child_idx]; // ref to pointer
            mutex.lock();
            if (child == nullptr)
            {
                child = new ChanceNode();
            }
            mutex.unlock();
            return child;
        };

        size_t count_matrix_nodes() const
        {
            size_t c = 1;
            const size_t n_children = rows * cols;

            for (size_t i = 0; i < n_children; ++i)
            {
                ChanceNode *&chance_node = edges[i];
                if (edges[i] != nullptr)
                {
                    c += chance_node->count_matrix_nodes();
                }
            }
            return c;
        }
    };

    class ChanceNode
    {
    public:
        std::unordered_map<
            typename Types::Obs,
            MatrixNode *,
            typename Types::ObsHash>
            edges{};
        ChanceStats stats{};

        ChanceNode() {}
        ChanceNode(const ChanceNode &) = delete;
        ~ChanceNode();

        MatrixNode *access(const Types::Obs &obs)
        {
            MatrixNode *&child = edges[obs];
            if (child == nullptr)
            {
                child = new MatrixNode(obs);
            }
            return child;
        };

        const MatrixNode *access(const Types::Obs &obs) const
        {
            return edges[obs];
        };

        MatrixNode *access(const Types::Obs &obs, Types::Mutex &mutex)
        {
            MatrixNode *&child = edges[obs];
            mutex.lock();
            if (child == nullptr)
            {
                child = new MatrixNode(obs);
            }
            mutex.unlock();
            return child;
        };

        size_t count_matrix_nodes() const
        {
            size_t c = 0;
            for (const auto &[obs, matrix_node] : edges)
            {
                if (matrix_node != nullptr)
                {
                    c += matrix_node->count_matrix_nodes();
                }
            }
            return c;
        }
    };
};

template <IsStateTypes Types, typename MStats, typename CStats,
          typename stores_actions, typename stores_value>
FlatNodes<Types, MStats, CStats, stores_actions, stores_value>::MatrixNode::~MatrixNode()
{
    delete[] edges;
}

template <IsStateTypes Types, typename MStats, typename CStats,
          typename stores_actions, typename stores_value>
FlatNodes<Types, MStats, CStats, stores_actions, stores_value>::ChanceNode::~ChanceNode()
{
    for (const auto &[obs, matrix_node] : edges)
    {
        delete matrix_node;
    }
};
