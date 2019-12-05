## 必要なライブラリ
cv2  
OpenJTalk
```
$ sudo apt-get install open-jtalk open-jtalk-mecab-naist-jdic hts-voice-nitech-jp-atr503-m001
```
また、音声データも必要
```
$ wget https://sourceforge.net/projects/mmdagent/files/MMDAgent_Example/MMDAgent_Example-1.6/MMDAgent_Example-1.6.zip
$ unzip ./MMDAgent_Example-1.6.zip 
$ sudo cp -R ./MMDAgent_Example-1.6/Voice/mei /usr/share/hts-voice/

```
yoloの学習済みデータダウンロード
```
$ cd pytorch_yolo_v3
$ wget https://pjreddie.com/media/files/yolov3.weights

```

## 使い方
pytorh_yolo_v3/video_demo.py をpython3系で実行

## 自分用のtrainingデータを作るには
```
self_images/<各ラベル名>/<pngデータ>
```
に画像データを入れる。
validationは
```
self_images_val/<各ラベル名>/<pngデータ>
```
に画像を入れる。( config.py のラベルも合わせる)
あとは MakeDataset.py を実行したあと、 TrainMain.py を実行するだけ。

Augumentation.py を実行させることでself_images/ 内の画像データを水増しできる。

## 備考
- video_demo.pyのdef writeのコメント文を直接いじってハミング距離での推定かCNNかを選択できる(デフォルトはCNN)

- gpuがない場合は config.py の設定を変える必要がある。

- その他細かい設定をconfigで大体かえることができる
