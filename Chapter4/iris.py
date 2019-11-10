from sklearn.datasets import load_iris
from sklearn import tree

def main():
  # アヤメのデータを読み込む
  iris = load_iris()

  # アヤメの素性(説明変数)はリスト
  # 順にがく片の幅，がく片の長さ，花弁の幅，花弁の長さ
  # print(iris.data)

  # アヤメの種類(目的変数)は3種類(3値分類)
  # print(iris.target)

  '''
    今回の内容と関係ありそうなパラメータ
    criterion = 'gini' or 'entropy' (default: 'gini')                        # 分割する際にどちらを使うか
    max_depth = INT_VAL or None (default: None)                              # 作成する決定木の最大深さ
    min_samples_split = INT_VAL (default: 2)                                 # サンプルを分割する際の枝の数の最小値
    min_samples_leaf = INT_VAL (default: 1)                                  # 1つのサンプルが属する葉の数の最小値
    min_weight_fraction_leaf = FLOAT_VAL (default: 0.0)                      # 1つの葉に属する必要のあるサンプルの割合の最小値
    max_leaf_nodes = INT_VAL or None (default: None)                         # 作成する葉の最大値(設定するとmax_depthが無視される)
    class_weight = DICT, LIST_OF_DICTS, 'balanced', or None (default: None)  # 各説明変数に対する重み
    presort = BOOL (default: False)                                          # 高速化のための入力データソートを行うか
  '''
  # モデルを作成
  clf = tree.DecisionTreeClassifier(max_depth = 3)
  clf = clf.fit(iris.data, iris.target)

  # 作成したモデルを用いて予測を実行
  predicted = clf.predict(iris.data)

  # 予測結果の出力(正解データとの比較)
  print('=============== predicted ===============')
  print(predicted)
  print('============== correct_ans ==============')
  print(iris.target)
  print('=============== id_rate =================')
  print(sum(predicted == iris.target) / len(iris.target))

  '''
    feature_namesには各説明変数の名前を入力
    class_namesには目的変数の名前を入力
    filled = Trueで枝に色が塗られる
    rounded = Trueでノードの角が丸くなる
  '''
  # 学習したモデルの可視化
  # これによって同じディレクトリにiris_model.dotが出力されるので中身をwww.webgraphviz.comに貼り付けたら可視化できる
  f = tree.export_graphviz(clf, out_file = 'iris_model.dot', feature_names = iris.feature_names,
                           class_names = iris.target_names, filled = True, rounded = True)

if __name__ == '__main__':
  main()
