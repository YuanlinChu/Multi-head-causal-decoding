import numpy as np
import math
from typing import List, Tuple, Dict, Set, Optional
import random

class MCTSNode:
    def __init__(self, state=None, parent=None, action=None):
        self.state = state if state else (0, [])  # (current_node, path_so_far)
        self.parent = parent
        self.action = action  # 导致这个状态的动作
        self.children = {}  # 子节点字典 {action: node}
        self.visits = 0  # 访问次数
        self.reward = 0.0  # 累计奖励
        self.untried_actions = self._get_untried_actions()  # 未尝试的动作
    
    def _get_untried_actions(self) -> List[int]:
        """获取当前节点可用的未尝试动作"""
        current_node, _ = self.state
        
        # 如果已经到达节点4，只能选择0-4中的一个选项，然后终止
        if current_node == 4:
            return list(range(5))
        
        # 其他节点可以选择0-4或者停止(-1)
        return list(range(5)) + [-1]  # -1表示停止
    
    def is_fully_expanded(self) -> bool:
        """检查是否所有可能的动作都已经被尝试"""
        return len(self.untried_actions) == 0
    
    def is_terminal(self) -> bool:
        """检查当前节点是否是终止节点"""
        current_node, path = self.state
        # 如果路径长度已达到5，或者选择了停止，或者已经到达节点4并做出选择
        return len(path) >= 5 or (path and path[-1] == -1) or current_node >= 5
    
    def get_ucb(self, exploration_weight: float) -> float:
        """计算UCB值"""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.reward / self.visits
        exploration = exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration
    
    def best_child(self, exploration_weight: float) -> 'MCTSNode':
        """选择UCB值最高的子节点"""
        return max(self.children.values(), key=lambda child: child.get_ucb(exploration_weight))
    
    def expand(self, action: int, prior_probs: np.ndarray) -> 'MCTSNode':
        """扩展一个新的子节点"""
        current_node, path = self.state
        
        # 如果选择停止
        if action == -1:
            new_state = (current_node, path + [action])
        else:
            # 如果选择继续，移动到下一个节点
            new_state = (current_node + 1, path + [action])
        
        child = MCTSNode(state=new_state, parent=self, action=action)
        self.untried_actions.remove(action)
        self.children[action] = child
        return child

class MCTS:
    def __init__(self, exploration_weight: float = math.sqrt(2)):
        self.exploration_weight = exploration_weight
        self.prior_probs = np.ones((5, 5)) * 0.2  # 初始化每个节点每个选项的概率为0.2
    
    def search(self, root: MCTSNode, num_simulations: int) -> MCTSNode:
        """执行MCTS搜索"""
        for _ in range(num_simulations):
            node = self._select(root)
            if not node.is_terminal():
                node = self._expand(node)
            reward = self._simulate(node)
            self._backpropagate(node, reward)
        return root
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """选择阶段：选择最有前途的节点进行扩展"""
        while not node.is_terminal() and node.is_fully_expanded():
            node = node.best_child(self.exploration_weight)
        return node
    
    def _expand(self, node: MCTSNode) -> MCTSNode:
        """扩展阶段：从未尝试的动作中选择一个并创建新节点"""
        action = random.choice(node.untried_actions)
        return node.expand(action, self.prior_probs)
    
    def _simulate(self, node: MCTSNode) -> float:
        """模拟阶段：从当前节点模拟到终止状态并返回奖励"""
        current_node, path = node.state
        total_reward = 0.0
        
        # 如果已经是终止状态，直接返回0
        if node.is_terminal():
            return 0.0
        
        # 模拟直到终止
        while current_node < 5 and len(path) < 5:
            # 如果选择停止，终止模拟
            if path and path[-1] == -1:
                break
            
            # 随机选择一个动作（包括停止）
            action = random.choice(list(range(5)) + [-1])
            
            # 如果选择停止
            if action == -1:
                path.append(action)
                break
            
            # 如果选择继续，计算是否成功前进
            success = random.random() < self.prior_probs[current_node, action]
            
            # 每次选择都有资源成本
            total_reward -= 0.1
            
            if success:
                # 成功前进到下一个节点
                total_reward += 1.0  # 奖励
                current_node += 1
                path.append(action)
            else:
                # 选择错误，终止路径
                path.append(action)
                break
        
        return total_reward
    
    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        """回传阶段：更新节点及其祖先的统计信息"""
        while node:
            node.visits += 1
            node.reward += reward
            node = node.parent
    
    def generate_diverse_paths(self, num_paths: int = 8, max_path_length: int = 5) -> List[List[int]]:
        """生成多样化的路径"""
        root = MCTSNode()
        self.search(root, 10000)  # 执行10000次模拟
        
        # 从搜索树中提取路径
        paths = []
        candidate_nodes = self._get_promising_nodes(root)
        
        # 按照奖励排序
        candidate_nodes.sort(key=lambda n: n.reward / max(1, n.visits), reverse=True)
        
        # 选择多样化的路径
        selected_paths = set()
        for node in candidate_nodes:
            _, path = node.state
            
            # 跳过空路径
            if not path:
                continue
            
            # 检查路径是否是已选路径的前缀或者已选路径是否是该路径的前缀
            is_prefix = False
            for selected_path in selected_paths:
                min_len = min(len(path), len(selected_path))
                if path[:min_len] == selected_path[:min_len]:
                    is_prefix = True
                    break
            
            if not is_prefix and len(path) <= max_path_length:
                selected_paths.add(tuple(path))
                if len(selected_paths) >= num_paths:
                    break
        
        # 如果没有足够的路径，添加一些随机路径
        while len(selected_paths) < num_paths:
            path_length = random.randint(1, max_path_length)
            new_path = tuple([random.randint(0, 4) for _ in range(path_length)])
            
            # 检查是否与已有路径冲突
            is_prefix = False
            for selected_path in selected_paths:
                min_len = min(len(new_path), len(selected_path))
                if new_path[:min_len] == selected_path[:min_len]:
                    is_prefix = True
                    break
            
            if not is_prefix:
                selected_paths.add(new_path)
        
        return [list(path) for path in selected_paths]
    
    def _get_promising_nodes(self, root: MCTSNode) -> List[MCTSNode]:
        """获取有前途的节点列表"""
        promising_nodes = []
        
        def collect_nodes(node):
            if node.is_terminal():
                promising_nodes.append(node)
            else:
                for child in node.children.values():
                    collect_nodes(child)
        
        collect_nodes(root)
        return promising_nodes
    
    def update_prior_probs(self, paths: List[List[int]], feedback: List[int]) -> None:
        """根据反馈更新先验概率"""
        for path, actual_length in zip(paths, feedback):
            for i, action in enumerate(path):
                if action == -1:  # 跳过停止动作
                    continue
                
                node_idx = i  # 当前节点索引
                
                # 如果路径在该选择后继续前进
                if i < actual_length:
                    self.prior_probs[node_idx, action] += 0.05
                else:
                    # 如果路径在该选择处终止
                    self.prior_probs[node_idx, action] -= 0.05
                
                # 确保概率在[0.05, 0.95]范围内
                self.prior_probs[node_idx, action] = max(0.05, min(0.95, self.prior_probs[node_idx, action]))
            
            # 归一化概率
            for i in range(5):
                self.prior_probs[i] = self.prior_probs[i] / self.prior_probs[i].sum()

def main():
    # 创建MCTS实例
    mcts = MCTS()
    
    # 模拟多轮游戏
    num_rounds = 5
    for round_idx in range(num_rounds):
        print(f"Round {round_idx + 1}:")
        
        # 生成8条路径
        paths = mcts.generate_diverse_paths()
        print("生成的路径:")
        print(paths)  # 直接输出嵌套列表形式
        # for i, path in enumerate(paths):
        #     print(f"路径 {i+1}: {path}")
        
        # 模拟反馈（在实际应用中，这应该来自环境）
        # 这里我们随机生成反馈作为示例
        feedback = []
        for path in paths:
            # 模拟每条路径的实际长度
            actual_length = 0
            for i, action in enumerate(path):
                if action == -1:  # 如果选择停止
                    actual_length = i
                    break
                
                # 使用先验概率模拟是否成功前进
                success = random.random() < mcts.prior_probs[i, action]
                if success:
                    actual_length = i + 1
                else:
                    actual_length = i
                    break
            
            feedback.append(actual_length)
        
        print("反馈（实际长度）:", feedback)
        
        # 更新先验概率
        mcts.update_prior_probs(paths, feedback)
        print("更新后的先验概率:")
        for i in range(5):
            print(f"节点 {i}: {mcts.prior_probs[i]}")
        print()

if __name__ == "__main__":
    main()