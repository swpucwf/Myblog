# docsify & gitee 搭建个人博客

[TOC]



## 1.npm 安装

npm是Node.js的包管理器，用于安装和管理JavaScript包。要安装npm，需要先安装Node.js。以下是在不同操作系统上安装npm的步骤：

### 1.1 在Windows上安装npm：

1. 访问Node.js官方网站（https://nodejs.org）。
2. 在下载页面上，选择适用于Windows的LTS版本（长期支持版本）的Node.js安装程序。
3. 下载安装程序并运行它。
4. 在安装过程中，确保选中"npm package manager"选项。
5. 完成安装后，打开命令提示符或PowerShell，并运行npm -v命令，确认npm已成功安装。

### 1.2 在macOS上安装npm：

1. 打开终端应用程序。
2. 使用Homebrew包管理器安装Node.js。运行以下命令：

```shell
  brew install node
```

3. 完成安装后，运行npm -v命令，确认npm已成功安装。

### 1.3 linux 安装npm

**在Linux上安装npm：**

1. 打开终端。
2. 使用包管理器安装Node.js。根据你使用的Linux发行版，运行以下命令之一：

- Debian/Ubuntu：

```shell
	sudo apt-get install nodejs npm
```

- Fedora：

````shell
    sudo dnf install nodejs npm
````

- CentOS/RHEL：

```shell
     sudo yum install nodejs npm
```

安装完成后，你可以使用npm install <package-name>命令来安装JavaScript包。例如，要安装名为"lodash"的包，可以运行npm install lodash命令。

## 2. docsify

### 2.1 安装docsify

```shell
# 安装 docsify-cli
npm i docsify-cli -g
# 初始化项目
docsify init ./docs
# 发动项目
docsify serve docs
```

![image-20230812092252353](https://img-blog.csdnimg.cn/img_convert/03dcc1fa290645a5e616e68d1e248358.png)

![img](https://img-blog.csdnimg.cn/img_convert/620c693d3fc8ff73515765bbf561bf99.png)

![image-20230812092420688](https://img-blog.csdnimg.cn/img_convert/6cc8731b2c42a6b37436520586832f98.png)

http://localhost:3000/#/ 成功截图：

![image-20230812092559318](https://img-blog.csdnimg.cn/img_convert/fd8656c21826a0bcf7affbf8b1d85e57.png)

### 2.2 自定义配置

#### 2.2.1 通过修改index.html，定制化开发页面

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Document</title>
  <meta http-equiv="X-UA-Compatible" content="IE=edge,chrome=1" />
  <meta name="description" content="Description">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, minimum-scale=1.0">
  <link rel="stylesheet" href="//cdn.jsdelivr.net/npm/docsify@4/lib/themes/vue.css">
</head>
<body>
  <div id="app"></div>
  <script>
    window.$docsify = {
      name: '',
      repo: ''
    }
  </script>
  <!-- Docsify v4 -->
  <script src="//cdn.jsdelivr.net/npm/docsify@4"></script>
</body>
</html>
```

### 2.2.2 顶部导航栏和侧边栏

```javascript
   window.$docsify = {
       name: '东小西', // 侧边栏顶部显现的称号
       repo: 'https://gitee.com/chen_1953981601', // 右上角Github图标链接,这是例子，需要具体换为自己的
       loadNavbar: true, // 默许加载 _navbar.md，作为顶部导航栏
       loadSidebar: true // 默许加载 _sidebar.md，作为侧边栏
    }
```

#### 2.2.3 新建 _navbar.md 文件

```markdown
- [**目录**](README.md)
  - [**环境安装**](source/env/env.md)
  - [**编程相关学习**](source/books/books.md)
  - [**人工智能**](source/DL/DL.md)
  - [**工程部署相关问题**](source/project/project.md)
  - [**相关开源工具**](source/openTech/openTech.md)
  - **论文阅读笔记**
* [**文章集合**](paper/README.md)
```

#### 2.2.4 侧边栏设置

```javascript
 window.$docsify = {
       name: '东小西', // 侧边栏顶部显现的称号
       repo: 'https://gitee.com/renxiaoshi', // 右上角Github图标链接
       loadNavbar: true, // 默许加载 _navbar.md，作为顶部导航栏
	   loadSidebar: true, // 默许加载 _sidebar.md，作为侧边栏
	   subMaxLevel: 3, // 目录的最大层级
    }
```

#### 2.2.5 全文检索

```javascript
  window.$docsify = {
       name: '东小西', // 侧边栏顶部显现的称号
       repo: 'https://gitee.com/c_1953981601', // 右上角Github图标链接
       loadNavbar: true, // 默许加载 _navbar.md，作为顶部导航栏
	   loadSidebar: true, // 默许加载 _sidebar.md，作为侧边栏
	   subMaxLevel: 3, // 目录的最大层级
	   search: {
        paths: 'auto',
        placeholder: '检索',
        noData: '没有找到喔！',
        depth: 3,
      },
    }
  <!-- 检索插件 -->
  <script src="//cdn.jsdelivr.net/npm/docsify/lib/plugins/search.min.js"></script>
```

#### 2.2.6 一键仿制插件

```javascript
 <!-- 一键仿制插件 -->
  <script src="//cdn.jsdelivr.net/npm/docsify-copy-code"></script>
```

#### 2.2.7 代码高亮

```javascript
<!-- 代码高亮 -->
  <script src="//cdn.jsdelivr.net/npm/prismjs@1/components/prism-bash.min.js"></script>
  <script src="//cdn.jsdelivr.net/npm/prismjs@1/components/prism-c.min.js"></script>
  <script src="//cdn.jsdelivr.net/npm/prismjs@1/components/prism-json.min.js"></script>
  <script src="//cdn.jsdelivr.net/npm/prismjs@1/components/prism-shell-session.min.js"></script>
  <script src="//cdn.jsdelivr.net/npm/prismjs@1/components/prism-python.min.js"></script>
  <script src="//cdn.jsdelivr.net/npm/prismjs@1/components/prism-http.min.js"></script>
  <script src="//cdn.jsdelivr.net/npm/prismjs@1/components/prism-java.min.js"></script>
```

#### 2.2.8 代码高亮

```javascript
<link rel="stylesheet" href="//cdn.jsdelivr.net/npm/docsify/themes/dark.css">
```

#### 2.2.9 自定义封面

window.$docsify 中添加 coverpage: true，默许会加载 _coverpage.md。

新建_coverpage.md文件，内容如下：

```javascript
# Blogs for SWPUCWF


> 如果不是为了让她哭，那么卷人又有什么意义？

[CSDN](https://blog.csdn.net/weixin_42917352?spm=1000.2115.3001.5343)

email: swpucwf@126.com

[滚动鼠标](#)
```

## 3.gitee 搭建库

Gitee Pages服务，代码托管网站将用户的库房文件以网页方式发布。

![image-20230812105527551](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20230812105527551.png)

## 4 设置Github Pages

在 `Settings` 中的 `GitHub Pages` 中选择 `docs` 文件夹，点击保存，即可发布刚刚的文档网站。