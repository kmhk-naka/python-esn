# python-esn
シンプルなEcho State Networkで入力の1ステップ先を予測する．

## 内部状態の更新
<img src="https://render.githubusercontent.com/render/math?math=\boldsymbol{x}_{n%2B1} = \tanh (\boldsymbol{W}\boldsymbol{x}_n %2B \boldsymbol{W}^\mathrm{in}\boldsymbol{u}_{n%2B1})">

## 出力重みの更新
<img src="https://render.githubusercontent.com/render/math?math=W^\mathrm{out} = \left(X^T X %2B \lambda I\right)^{-1} X^T Y^\mathrm{target}">

# sin波
![sinusoid](https://raw.githubusercontent.com/kmhk-naka/python-esn/master/images/sinusoid.png)

# ロジスティック写像
`train_length = 2_000_000`で実行したときの結果

![logistic 2000000](https://raw.githubusercontent.com/kmhk-naka/python-esn/master/images/logistic-2000000.png)

![logistic feature 2000000](https://raw.githubusercontent.com/kmhk-naka/python-esn/master/images/logistic-feature-2000000.png)
