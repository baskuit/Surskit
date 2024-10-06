#pragma once

#include <libpinyon/math.h>
#include <state/state.h>
#include <tree/node.h>

/*

Identification of a recently transitioned state with the search tree
is done with a linear scan and equality check

This has two advantages (although there are also downsides)
First, the most relevent matrix nodes are usually the earliest in the chance node's children
(and policy guided search will put the most relevent joint actions first, as well)
Second, it's totally general in that we don't need to implement a hash function
(and we also don't have to deal with hash collisions)

*/

template <IsStateTypes Types, typename MStats, typename CStats, typename NodeActions = void,
          typename NodeValue = void>
struct DefaultNodes : Types {
    friend std::ostream &operator<<(std::ostream &os, const DefaultNodes &) {
        os << "DefaultNodes";
        return os;
    }

    class MatrixNode;

    class ChanceNode;

    using MatrixStats = MStats;
    using ChanceStats = CStats;

    class MatrixNode : public MatrixNodeData<Types, NodeActions, NodeValue> {
       public:
        ChanceNode *child = nullptr;
        MatrixNode *next = nullptr;

        bool terminal = false;
        bool expanded = false;

        Types::Obs obs;
        MatrixStats stats;

        MatrixNode(){};
        MatrixNode(Types::Obs obs) : obs(obs) {}
        MatrixNode(const MatrixNode &) = delete;
        ~MatrixNode();

        inline void expand(const size_t &, const size_t &) { expanded = true; }

        inline bool terminal() const { return terminal; }

        inline bool is_expanded() const { return expanded; }

        inline void set_terminal() { terminal = true; }

        inline void set_expanded() { expanded = true; }

        inline void get_value(Types::Value &value) const {}

        ChanceNode *access(int row_idx, int col_idx) {
            if (this->child == nullptr) {
                this->child = new ChanceNode(row_idx, col_idx);
                return this->child;
            }
            ChanceNode *current = this->child;
            ChanceNode *previous = this->child;
            while (current != nullptr) {
                previous = current;
                if (current->row_idx == row_idx && current->col_idx == col_idx) {
                    return current;
                }
                current = current->next;
            }
            ChanceNode *child = new ChanceNode(row_idx, col_idx);
            previous->next = child;
            return child;
        };

        const ChanceNode *access(int row_idx, int col_idx) const {
            if (this->child == nullptr) {
                return this->child;
            }
            const ChanceNode *current = this->child;
            const ChanceNode *previous = this->child;
            while (current != nullptr) {
                previous = current;
                if (current->row_idx == row_idx && current->col_idx == col_idx) {
                    return current;
                }
                current = current->next;
            }
            return current;
        };

        ChanceNode *access(int row_idx, int col_idx, Types::Mutex &mutex) {
            mutex.lock();
            if (this->child == nullptr) {
                this->child = new ChanceNode(row_idx, col_idx);
                mutex.unlock();
                return this->child;
            }
            ChanceNode *current = this->child;
            ChanceNode *previous = this->child;
            while (current != nullptr) {
                previous = current;
                if (current->row_idx == row_idx && current->col_idx == col_idx) {
                    mutex.unlock();
                    return current;
                }
                current = current->next;
            }
            ChanceNode *child = new ChanceNode(row_idx, col_idx);
            previous->next = child;
            mutex.unlock();
            return child;
        };

        size_t count_matrix_nodes() const {
            size_t c = 1;
            ChanceNode *current = this->child;
            while (current != nullptr) {
                c += current->count_matrix_nodes();
                current = current->next;
            }
            return c;
        }
    };

    class ChanceNode {
       public:
        MatrixNode *child = nullptr;
        ChanceNode *next = nullptr;

        int row_idx;
        int col_idx;

        ChanceStats stats;

        ChanceNode() {}
        ChanceNode(int row_idx, int col_idx) : row_idx(row_idx), col_idx(col_idx) {}
        ChanceNode(const ChanceNode &) = delete;

        ~ChanceNode();

        MatrixNode *access(const Types::Obs &obs) {
            if (this->child == nullptr) {
                MatrixNode *child = new MatrixNode(obs);
                this->child = child;
                return child;
            }
            MatrixNode *current = this->child;
            MatrixNode *previous = this->child;
            while (current != nullptr) {
                previous = current;
                if (current->obs == obs) {
                    return current;
                }
                current = current->next;
            }
            MatrixNode *child = new MatrixNode(obs);
            previous->next = child;
            return child;
        };

        const MatrixNode *access(const Types::Obs &obs) const {
            if (this->child == nullptr) {
                return this->child;
            }
            const MatrixNode *current = this->child;
            const MatrixNode *previous = this->child;
            while (current != nullptr) {
                previous = current;
                if (current->obs == obs) {
                    return current;
                }
                current = current->next;
            }
            return current;
        };

        MatrixNode *access(const Types::Obs &obs, Types::Mutex &mutex) {
            mutex.lock();
            if (this->child == nullptr) {
                MatrixNode *child = new MatrixNode(obs);
                this->child = child;
                mutex.unlock();
                return child;
            }
            MatrixNode *current = this->child;
            MatrixNode *previous = this->child;
            while (current != nullptr) {
                previous = current;
                if (current->obs == obs) {
                    mutex.unlock();
                    return current;
                }
                current = current->next;
            }
            MatrixNode *child = new MatrixNode(obs);
            previous->next = child;
            mutex.unlock();
            return child;
        };

        size_t count_matrix_nodes() const {
            size_t c = 0;
            MatrixNode *current = this->child;
            while (current != nullptr) {
                c += current->count_matrix_nodes();
                current = current->next;
            }
            return c;
        }
    };
};

// We have to hold off on destructor definitions until here
template <IsStateTypes Types, typename MStats, typename CStats, typename stores_actions,
          typename stores_value>
DefaultNodes<Types, MStats, CStats, stores_actions, stores_value>::MatrixNode::~MatrixNode() {
    while (this->child != nullptr) {
        DefaultNodes<Types, MatrixStats, ChanceStats, stores_actions, stores_value>::ChanceNode *victim = this->child;
        this->child = this->child->next;
        delete victim;
    }
}
template <IsStateTypes Types, typename MStats, typename CStats, typename stores_actions,
          typename stores_value>
DefaultNodes<Types, MStats, CStats, stores_actions, stores_value>::ChanceNode::~ChanceNode() {
    while (this->child != nullptr) {
        DefaultNodes<Types, MatrixStats, ChanceStats, stores_actions, stores_value>::MatrixNode *victim = this->child;
        this->child = this->child->next;
        delete victim;
    }
};
