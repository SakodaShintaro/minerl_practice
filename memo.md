意外と拡散モデルでの次状態予測を書く上で考えるべきことが多い。入力のタイムステップをどれくらいにするか、予測のタイムステップをどれくらいにするか。
ニューラルネットワークのモデルもタイムステップが入るので画像部分でいろいろいじる必要がある。
また、行動の埋め込みも同様にタイムステップがあるのでこれをどうすると綺麗なのかは不明。

画像は 360 * 640が 1/8ずつされて (45, 80)になる。
actionのdimは24
