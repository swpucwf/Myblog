[TOC]

# ubuntu安装搜狗输入法

## 1.添加中文语言支持

1. 打开 系统设置——区域和语言——管理已安装的语言——在“语言”tab下——点击“添加或删除语言”
2. 弹出“已安装语言”窗口，在中文（简体）方框中勾选，点击应用。

![image-20221231160838953](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221231160838953.png)

3. 回到“语言支持”窗口，在键盘输入法系统中，选择“fcitx”

```shell
sudo apt-get install fcitx
```

![image-20221231165238781](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221231165238781.png)

```shell
sudo dpkgg -i sogoupinyin_4.0.1.2800_x86_64.deb 
```

![image-20221231171157596](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20221231171157596.png)

## 解决Ubuntu搜狗输入法无法使用的问题

1.先卸载掉fcitx，及其所有相关的软件

```shell
sudo apt -y --purge remove fcitx
sudo apt clean fcitx
sudo apt -y install fcitx fcitx-bin fcitx-table fcitx-table-all
sudo apt -y install fcitx-config-gtk
sudo apt -y install fcitx-libs libfcitx-qt0 libopencc2 libopencc2-data libqt4-opengl libqtwebkit4
wget http://cdn2.ime.sogou.com/dl/index/1571302197/sogoupinyin_2.3.1.0112_amd64.deb
sudo dpkg -i sogoupinyin_2.3.1.0112_amd64.deb
 如果安装失败，请执行如下命令安装依赖，然后再执行上面的安装命令
sudo apt -f install
```

## 在处理时有错误发生: sogoupinyin

```shell
sudo add-apt-repository ppa:fcitx-team/nightly

sudo apt-get update

sudo apt-get upgrade

sudo apt install libopencc1 fcitx-libs fcitx-libs-qt fonts-droid-fallback

sudo apt-get install -f

sudo dpkg -i sogoupinyin_2.3.1.0112_amd64.deb

```





