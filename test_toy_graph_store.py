import torch
from test_graph_store import ToyGraphStore  # 从当前文件导入

# 创建ToyGraphStore实例
gs = ToyGraphStore()

print("== 测试边的存储和读取 ==")

# 创建一些测试数据
# 论文引用论文的边（共4条）
paper_cites_paper = torch.tensor([
    [0, 1, 2, 3],  # 源节点ID
    [1, 2, 3, 0]   # 目标节点ID
])

# 存储边
gs.put_edge_index(paper_cites_paper, ('paper', 'cites', 'paper'))
print("已存储 paper-cites->paper 边")

# 读取边
retrieved_edge_index = gs.get_edge_index(('paper', 'cites', 'paper'))
print("读取到的边:")
print(retrieved_edge_index)

# 测试语法糖
print("\n== 测试语法糖 ==")
sugar_retrieved = gs[('paper', 'cites', 'paper'), 'coo']
print("使用语法糖读取到的边:")
print(sugar_retrieved)
print("语法糖方式和普通方式结果是否相同:", torch.equal(retrieved_edge_index, sugar_retrieved))

# 测试不存在的边类型
print("\n== 测试不存在的边类型 ==")
nonexistent = gs.get_edge_index(('author', 'writes', 'paper'))
print(f"不存在的边类型返回值: {nonexistent}")

# 测试边的追加
print("\n== 测试边的追加 ==")
# 再添加两条论文引用论文的边
more_citations = torch.tensor([
    [4, 5],  # 源节点ID
    [5, 0]   # 目标节点ID
])
gs.put_edge_index(more_citations, ('paper', 'cites', 'paper'), mode='append')
print("追加后的所有 paper-cites->paper 边:")
appended_edges = gs.get_edge_index(('paper', 'cites', 'paper'))
print(appended_edges)
print(f"边数量: {appended_edges.shape[1]}")  # 应该是4+2=6条边

# 测试边的覆盖
print("\n== 测试边的覆盖 ==")
replacement_edges = torch.tensor([
    [10, 11],  # 源节点ID
    [11, 10]   # 目标节点ID
])
gs.put_edge_index(replacement_edges, ('paper', 'cites', 'paper'), mode='overwrite')
print("覆盖后的 paper-cites->paper 边:")
overwritten_edges = gs.get_edge_index(('paper', 'cites', 'paper'))
print(overwritten_edges)
print(f"边数量: {overwritten_edges.shape[1]}")  # 应该是2条边

print("\n测试完成!")
