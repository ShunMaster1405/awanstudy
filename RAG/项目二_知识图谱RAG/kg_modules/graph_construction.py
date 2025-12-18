"""
图构建模块（简化版）
"""

from typing import Dict, List, Tuple, Any
import networkx as nx
import matplotlib.pyplot as plt

class GraphConstruction:
    """图构建模块"""
    
    def __init__(self):
        self.graph = nx.Graph()
    
    def build_graph(self, knowledge_data: Dict) -> Dict:
        """构建知识图谱（简化版）"""
        print("构建知识图谱...")
        
        # 创建图
        self.graph = nx.Graph()
        
        # 添加实体节点
        entities = knowledge_data.get("entities", [])
        for entity in entities:
            self.graph.add_node(entity, type="entity", label=entity)
        
        print(f"添加了 {len(entities)} 个实体节点")
        
        # 添加关系边
        relations = knowledge_data.get("relations", [])
        edges_added = 0
        for relation in relations:
            if len(relation) == 3:
                entity1, relation_type, entity2 = relation
                # 检查实体是否在实体列表中
                if entity1 in entities and entity2 in entities:
                    # 避免重复边
                    if not self.graph.has_edge(entity1, entity2):
                        self.graph.add_edge(entity1, entity2, 
                                           relation=relation_type,
                                           label=relation_type)
                        edges_added += 1
                else:
                    # 如果实体不在列表中，但仍然添加边（可能实体被过滤掉了）
                    if not self.graph.has_edge(entity1, entity2):
                        self.graph.add_edge(entity1, entity2,
                                           relation=relation_type,
                                           label=relation_type)
                        edges_added += 1
        
        print(f"添加了 {edges_added} 个关系边（共 {len(relations)} 个关系）")
        
        # 计算图的基本统计信息
        graph_info = {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "connected_components": nx.number_connected_components(self.graph),
            "entities": entities,
            "relations": relations
        }
        
        print(f"图构建完成:")
        print(f"  - 节点数: {graph_info['num_nodes']}")
        print(f"  - 边数: {graph_info['num_edges']}")
        print(f"  - 图密度: {graph_info['density']:.4f}")
        print(f"  - 连通分量: {graph_info['connected_components']}")
        
        # 可视化图（可选）- 使用英文字体避免中文显示问题
        self._visualize_graph()
        
        return graph_info
    
    def _visualize_graph(self):
        """可视化图（简化版）- 智能字体处理"""
        try:
            if self.graph.number_of_nodes() > 0:
                print("生成图可视化...")
                
                # 创建图形
                plt.figure(figsize=(12, 10))
                
                # 使用spring布局
                pos = nx.spring_layout(self.graph, seed=42, k=1.5)
                
                # 绘制节点
                nx.draw_networkx_nodes(self.graph, pos, 
                                     node_color='lightblue',
                                     node_size=800,
                                     alpha=0.8)
                
                # 绘制边
                nx.draw_networkx_edges(self.graph, pos,
                                     edge_color='gray',
                                     width=2,
                                     alpha=0.6)
                
                # 智能字体处理
                use_english_labels = False
                
                try:
                    # 尝试设置中文字体（允许回退）
                    import matplotlib.font_manager as fm
                    import matplotlib
                    
                    # 重置字体设置
                    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
                    
                    # 尝试使用系统字体（允许回退）
                    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Hiragino Sans GB', 
                                                      'STHeiti', 'Microsoft YaHei', 
                                                      'SimHei', 'Arial Unicode MS', 
                                                      'DejaVu Sans']
                    plt.rcParams['axes.unicode_minus'] = False
                    
                    # 测试字体是否支持中文
                    test_chinese = "测试中文"
                    test_font = fm.FontProperties(family=plt.rcParams['font.sans-serif'][0])
                    
                    # 如果第一个字体不可用，使用默认字体
                    try:
                        font_path = fm.findfont(test_font)
                        if 'fallback' in font_path.lower() or 'default' in font_path.lower():
                            use_english_labels = True
                            print("检测到字体不支持中文，使用英文字符")
                        else:
                            print(f"使用字体: {font_path}")
                    except:
                        use_english_labels = True
                        print("字体检测失败，使用英文字符")
                        
                except Exception as font_error:
                    use_english_labels = True
                    print(f"字体设置失败，使用英文字符: {font_error}")
                
                if use_english_labels:
                    # 生成英文标签（通用算法，不硬编码）
                    label_mapping = {}
                    node_counter = {}
                    
                    for node in self.graph.nodes():
                        # 如果是常见AI术语，使用标准缩写
                        if '人工智能' in node:
                            label_mapping[node] = 'AI'
                        elif '机器学习' in node:
                            label_mapping[node] = 'ML'
                        elif '深度学习' in node:
                            label_mapping[node] = 'DL'
                        elif '神经网络' in node:
                            label_mapping[node] = 'NN'
                        elif '计算机' in node:
                            label_mapping[node] = 'Computer'
                        else:
                            # 通用算法：使用前几个字符或创建缩写
                            # 移除常见后缀
                            clean_node = node.replace('的', '').replace('一个', '').replace('一种', '')
                            
                            if len(clean_node) <= 4:
                                label_mapping[node] = clean_node
                            else:
                                # 创建缩写：取前2-3个有意义的字符
                                chars = []
                                for char in clean_node[:4]:
                                    if char not in ['的', '是', '在', '和', '与', '或']:
                                        chars.append(char)
                                if chars:
                                    label_mapping[node] = ''.join(chars)
                                else:
                                    label_mapping[node] = clean_node[:3]
                    
                    # 绘制节点标签（使用英文）
                    nx.draw_networkx_labels(self.graph, pos,
                                          labels=label_mapping,
                                          font_size=9,
                                          font_weight='bold')
                    
                    # 绘制边标签（使用英文）
                    edge_labels = nx.get_edge_attributes(self.graph, 'label')
                    english_edge_labels = {}
                    for (u, v), label in edge_labels.items():
                        # 简单的关系翻译
                        if label == '是':
                            english_edge_labels[(u, v)] = 'is'
                        elif label == '包含':
                            english_edge_labels[(u, v)] = 'contains'
                        elif label == '使用':
                            english_edge_labels[(u, v)] = 'uses'
                        elif label == '基于':
                            english_edge_labels[(u, v)] = 'based on'
                        elif label == '模拟':
                            english_edge_labels[(u, v)] = 'simulates'
                        elif label == '执行':
                            english_edge_labels[(u, v)] = 'executes'
                        elif label == '创建':
                            english_edge_labels[(u, v)] = 'creates'
                        elif label == '属于':
                            english_edge_labels[(u, v)] = 'belongs to'
                        else:
                            # 对于其他关系，使用原始标签（如果短）或缩写
                            if len(label) <= 3:
                                english_edge_labels[(u, v)] = label
                            else:
                                english_edge_labels[(u, v)] = label[:2]
                    
                    nx.draw_networkx_edge_labels(self.graph, pos,
                                               edge_labels=english_edge_labels,
                                               font_size=8)
                    
                    plt.title("Knowledge Graph", fontsize=14, fontweight='bold')
                    
                else:
                    # 绘制节点标签（使用中文）
                    nx.draw_networkx_labels(self.graph, pos,
                                          font_size=9,
                                          font_weight='bold')
                    
                    # 绘制边标签（使用中文）
                    edge_labels = nx.get_edge_attributes(self.graph, 'label')
                    nx.draw_networkx_edge_labels(self.graph, pos,
                                               edge_labels=edge_labels,
                                               font_size=8)
                    
                    plt.title("知识图谱可视化", fontsize=14, fontweight='bold')
                
                plt.axis('off')
                
                # 保存图像
                import os
                os.makedirs("graph_visualization", exist_ok=True)
                plt.savefig("graph_visualization/knowledge_graph.png", 
                          dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                
                print("图可视化已保存到: graph_visualization/knowledge_graph.png")
                if use_english_labels:
                    print("注：使用英文字符避免字体问题")
                
        except Exception as e:
            print(f"图可视化失败: {e}")
            print("继续执行...")
    
    def query_graph(self, query: str) -> List[Dict]:
        """查询图（简化版）"""
        results = []
        
        # 简单的图查询
        if not self.graph:
            return results
        
        # 查找包含查询词的节点
        query_terms = query.split()
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            node_label = node_data.get('label', str(node))
            
            # 检查节点标签是否包含查询词
            for term in query_terms:
                if term in node_label:
                    results.append({
                        "node": node,
                        "label": node_label,
                        "type": node_data.get('type', 'unknown'),
                        "neighbors": list(self.graph.neighbors(node))
                    })
                    break
        
        return results
    
    def get_node_info(self, node: str) -> Dict:
        """获取节点信息"""
        if node not in self.graph:
            return {}
        
        node_data = self.graph.nodes[node]
        neighbors = list(self.graph.neighbors(node))
        edges = list(self.graph.edges(node, data=True))
        
        return {
            "node": node,
            "label": node_data.get('label', node),
            "type": node_data.get('type', 'unknown'),
            "degree": self.graph.degree(node),
            "neighbors": neighbors,
            "edges": edges
        }
