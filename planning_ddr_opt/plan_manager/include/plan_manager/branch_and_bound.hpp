#ifndef BRANCH_AND_BOUND_COMBINED_HPP
#define BRANCH_AND_BOUND_COMBINED_HPP

#include <vector>
#include <queue>
#include <limits>
#include <algorithm>
#include <Eigen/Eigen>
#include "hungarian.hpp" // Note: This header is still included for historical/potential future use, though the fixed-assignment lower bound doesn't use it.

// Node for the combined Assignment-Routing B&B
struct CombinedNode {
    // State
    int last_pos_idx; // Global index in the distance matrix (0=start, 1..n=chairs, n+1..2n=targets)
    int visited_chairs_mask;
    int assigned_targets_mask;
    int level; // How many pairs have been visited

    // Path & Cost
    double current_cost;
    double lower_bound;
    std::vector<int> path_indices; // Sequence of global indices

    bool operator>(const CombinedNode& other) const {
        return lower_bound > other.lower_bound;
    }
};

class BranchAndBoundCombined {
public:
    BranchAndBoundCombined(const Eigen::MatrixXd& dists, int num_tasks) 
        : all_dists_(dists), num_tasks_(num_tasks) {}

    // Solves the routing problem with a fixed assignment
    double solve(const std::vector<int>& assignment, std::vector<int>& best_path_indices) {
        // --- Greedy pre-processing for a tight initial upper bound ---
        std::vector<int> greedy_path;
        double greedy_cost = solveGreedy(assignment, greedy_path); 
        double global_best_cost = greedy_cost;
        best_path_indices = greedy_path;

        std::priority_queue<CombinedNode, std::vector<CombinedNode>, std::greater<CombinedNode>> pq;

        CombinedNode root;
        root.last_pos_idx = 0;
        root.visited_chairs_mask = 0;
        root.assigned_targets_mask = 0;
        root.level = 0;
        root.current_cost = 0;
        root.path_indices.push_back(0);
        root.lower_bound = calculateLowerBound(root, assignment);
        
        pq.push(root);

        while (!pq.empty()) {
            CombinedNode current = pq.top();
            pq.pop();

            if (current.lower_bound >= global_best_cost) {
                continue;
            }

            if (current.level == num_tasks_) {
                if (current.current_cost < global_best_cost) {
                    global_best_cost = current.current_cost;
                    best_path_indices = current.path_indices;
                }
                continue;
            }

            // Branching: only considers unvisited chairs, and their fixed assigned targets
            for (int i = 0; i < num_tasks_; ++i) {
                if (!((current.visited_chairs_mask >> i) & 1)) {
                    CombinedNode child = current;
                    child.level++;

                    // Use the fixed assignment: chair 'i' goes to target 'assignment[i]'
                    int chair_idx = 1 + i;
                    int target_idx = 1 + num_tasks_ + assignment[i];
                    
                    child.current_cost += all_dists_(current.last_pos_idx, chair_idx);
                    child.current_cost += all_dists_(chair_idx, target_idx);
                    
                    child.last_pos_idx = target_idx;
                    child.visited_chairs_mask |= (1 << i);
                    child.assigned_targets_mask |= (1 << assignment[i]);
                    
                    child.path_indices.push_back(chair_idx);
                    child.path_indices.push_back(target_idx);

                    child.lower_bound = calculateLowerBound(child, assignment);
                    if (child.lower_bound < global_best_cost) {
                        pq.push(child);
                    }
                }
            }
        }
        return global_best_cost;
    }

private:
    const Eigen::MatrixXd& all_dists_;
    int num_tasks_;

    // Greedily finds an initial path with a fixed assignment
    double solveGreedy(const std::vector<int>& assignment, std::vector<int>& path) {
        double cost = 0.0;
        int current_pos = 0;
        std::vector<bool> visited_pairs(num_tasks_, false);

        path.clear();
        path.push_back(0);

        for (int k = 0; k < num_tasks_; ++k) {
            double min_dist = std::numeric_limits<double>::max();
            int next_chair_idx = -1;
            
            // Find the closest unvisited chair
            for (int i = 0; i < num_tasks_; ++i) {
                if (!visited_pairs[i]) {
                    double dist = all_dists_(current_pos, 1 + i);
                    if (dist < min_dist) {
                        min_dist = dist;
                        next_chair_idx = 1 + i;
                    }
                }
            }
            if (next_chair_idx == -1) break;
            
            // Use the fixed assignment to find the corresponding target
            int target_idx = 1 + num_tasks_ + assignment[next_chair_idx - 1];
            
            cost += min_dist + all_dists_(next_chair_idx, target_idx);
            current_pos = target_idx;
            path.push_back(next_chair_idx);
            path.push_back(target_idx);
            visited_pairs[next_chair_idx - 1] = true;
        }
        return cost;
    }

    // Calculates a lower bound with a fixed assignment
    // Calculates a correct lower bound with a fixed assignment
    double calculateLowerBound(const CombinedNode& node, const std::vector<int>& assignment) {
        double bound = node.current_cost;

        // Collect remaining tasks to visit
        std::vector<int> remaining_chair_indices;
        for (int i = 0; i < num_tasks_; ++i) {
            if (!((node.visited_chairs_mask >> i) & 1)) {
                remaining_chair_indices.push_back(i);
            }
        }
        
        if (remaining_chair_indices.empty()) {
            return bound;
        }

        // 1. Cost from current position to the closest unvisited chair
        double min_dist_to_next_chair = std::numeric_limits<double>::max();
        for (int chair_idx : remaining_chair_indices) {
            min_dist_to_next_chair = std::min(min_dist_to_next_chair, all_dists_(node.last_pos_idx, 1 + chair_idx));
        }
        bound += min_dist_to_next_chair;

        // 2. Sum of all fixed chair->target costs for remaining tasks
        double sum_of_remaining_pairs = 0.0;
        for (int chair_idx : remaining_chair_indices) {
            int target_idx = assignment[chair_idx];
            sum_of_remaining_pairs += all_dists_(1 + chair_idx, 1 + num_tasks_ + target_idx);
        }
        bound += sum_of_remaining_pairs;

        // 3. Lower bound for connecting the final point back to the beginning.
        // For a fixed-assignment TSP, this is the cost from the final target to the start point (not implemented in your problem).
        // Let's assume the task is to visit all chairs and their targets, not return to start.
        // If the path must return to the start, a min cost from any remaining target to the start should be added.
        
        return bound;
    }
};

#endif // BRANCH_AND_BOUND_COMBINED_HPP