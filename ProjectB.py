import pandas as pd
import time

# 数据加载
data = pd.read_csv('C:/file/vscode/SVW-PY/Test/Order.csv',encoding = 'gbk')
# 统一小写
# data['Item'] = data['Item'].str.lower()
# 去掉none项
# data = data.drop(data[data.Item == 'none'].index)

# 采用efficient_apriori工具包
def rule1():
	from efficient_apriori import apriori
	start = time.time()
	# 得到一维数组orders_series，并且将Transaction作为index, value为Item取值
	orders_series = data.set_index('Transaction')['Item']
	print("一维数组orders_series：",orders_series)
	# 将数据集进行格式转换
	transactions = []
	temp_index = 0
	for i, v in orders_series.items():
		if i != temp_index:
			temp_set = set()
			temp_index = i
			temp_set.add(v)
			transactions.append(temp_set)
		else:
			temp_set.add(v)
	
	# 挖掘频繁项集和频繁规则
	itemsets, rules = apriori(transactions, min_support=0.01,  min_confidence = 0.5)
	print('频繁项集：', itemsets)
	print('关联规则：', rules)
	end = time.time()
	print("用时：", end-start)


def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1
# 采用mlxtend.frequent_patterns工具包
def rule2():
	from mlxtend.frequent_patterns import apriori
	from mlxtend.frequent_patterns import association_rules
	pd.options.display.max_columns=100
	start = time.time()
	hot_encoded_df=data.groupby(['Transaction','Item'])['Item'].count().unstack().reset_index().fillna(0).set_index('Transaction')
	hot_encoded_df = hot_encoded_df.applymap(encode_units)
	frequent_itemsets = apriori(hot_encoded_df, min_support=0.02, use_colnames=True)
	rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.5)
	print("频繁项集：", frequent_itemsets)
	print("关联规则：", rules[ (rules['lift'] >= 1) & (rules['confidence'] >= 0.5) ])
	#print(rules['confidence'])
	end = time.time()
	print("用时：", end-start)


# rule1()
print('-'*100)
rule2()

"""
----------------------------------------------------------------------------------------------------
频繁项集：      support                itemsets
0   0.346083                 (公路自行车)
1   0.322441                    (头盔)
2   0.221218                 (山地自行车)
3   0.115343                    (帽子)
4   0.074443                    (手套)
..       ...                     ...
67  0.021316         (手套, 头盔, 车胎和内胎)
68  0.020937     (头盔, 水壶和水壶架, 车胎和内胎)
69  0.029810        (头盔, 骑行服, 车胎和内胎)
70  0.029647  (水壶和水壶架, 山地自行车, 旅行自行车)
71  0.021749   (旅行自行车, 山地自行车, 车胎和内胎)

[72 rows x 2 columns]
关联规则：         antecedents consequents  antecedent support  consequent support  \
6           (旅行自行车)     (山地自行车)            0.115938            0.221218
17      (头盔, 山地自行车)     (公路自行车)            0.075092            0.346083
21     (公路自行车, 挡泥板)     (山地自行车)            0.026672            0.221218
22     (山地自行车, 挡泥板)     (公路自行车)            0.048583            0.346083
29   (山地自行车, 车胎和内胎)     (公路自行车)            0.059890            0.346083
32     (骑行服, 山地自行车)     (公路自行车)            0.036734            0.346083
34      (头盔, 旅行自行车)     (山地自行车)            0.057239            0.221218
36   (旅行自行车, 山地自行车)        (头盔)            0.067410            0.322441
44      (手套, 车胎和内胎)        (头盔)            0.036626            0.322441
49  (水壶和水壶架, 旅行自行车)     (山地自行车)            0.042199            0.221218
54   (旅行自行车, 车胎和内胎)     (山地自行车)            0.028619            0.221218

     support  confidence      lift  leverage  conviction
6   0.067410    0.581428  2.628299  0.041762    1.860568
17  0.039277    0.523055  1.511356  0.013289    1.371052
21  0.026672    1.000000  4.520421  0.020771         inf
22  0.026672    0.548998  1.586318  0.009858    1.449919
29  0.031378    0.523939  1.513910  0.010652    1.373598
32  0.020071    0.546392  1.578788  0.007358    1.441590
34  0.036626    0.639887  2.892556  0.023964    2.162601
36  0.036626    0.543339  1.685079  0.014891    1.483723
44  0.021316    0.581979  1.804917  0.009506    1.620874
49  0.029647    0.702564  3.175885  0.020312    2.618318
54  0.021749    0.759924  3.435178  0.015417    3.243902
用时： 0.5019853115081787
"""
