# 2.创建 Django 项目和 APP

**命令**：

- **创建Django项目**
  - django-admin startproject name
- **创建子应用**
  - python manager.py startapp name

## 2.1 创建工程

在使用Flask框架时，项目工程目录的组织与创建是需要我们自己手动创建完成的。

在django中，项目工程目录可以借助django提供的命令帮助我们创建。

### 2.1.1 创建

创建工程的命令为：

```
django-admin startproject 工程名称
```

例如：想要在桌面的source目录中创建一个名为ezfy的项目工程，可执行如下命令：

```
cd ~/Desktop/source
django-admin startproject ezfy
```

### 2.2.2 工程目录说明

查看创建的工程目录，结构如下：

![image-20230813162958209](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20230813162958209.png)

- 与项目同名的目录，此处为ezfy。
- **settings.py**是项目的整体配置文件。
- **urls.py**是项目的URL配置文件。
- **wsgi.py**是项目与WSGI兼容的Web服务器入口。
- **manage.py**是项目管理文件，通过它管理项目。

### 2.2.3 运行内置开发服务器

在开发阶段，django提供了一个纯python编写的轻量级web服务器，仅在开发阶段使用。

运行服务器命令如下：

```shell
python manage.py runserver ip:端口
或：
python manage.py runserver
```

![image-20230813163044387](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20230813163044387.png)

## 2.3 新建应用

### 2.3.1 同级目录建立应用

1. 安装 django

   ```shell 
   pip install django==4.2
   ```

2. 创建项目

   ```shell
   django-admin startproject ezfy
   ```

   其中 `ezfy` 指的是你的项目名字(`projectname`) ，目录如图

   ![image-20230813150233179](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20230813150233179.png)

3. 新建一个demo应用

   ```shell
   django-admin startapp demo
   ```

   ![image-20230813164829780](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20230813164829780.png)

4. 创建完成后，需要到与工程名相同的文件夹下（这里是ezfy）的 `settings.py` 中INSTALLED_APPS进行注册。

   注册名方式1：

   ![image-20230813165108094](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20230813165108094.png)

   注册名方式2：直接与应用名字相同![image-20230813165231305](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20230813165231305.png)

### 2.3.2  在目录下新建应用步骤

很多时候在同级目录下新建应用会造成文件夹众多，一般在apps包下开发多种应用，也就是集中将应用放在一个包下，这里做个演示。

1. 安装 django

   ```shell 
   pip install django==4.2
   ```

2. 创建项目

   ```shell
   django-admin startproject ezfy
   ```

   其中 `ezfy` 指的是你的项目名字(`projectname`) ，目录如图

   ![image-20230813150233179](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20230813150233179.png)

3. 创建 app  

   **在指定路径下创建 app：**

   新建一个apps包：

   ![image-20230813163157988](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20230813163157988.png)

   * 先 cd 到指定路径apps

   ```shell
   cd .\apps\
   ```

   * 运行 

   ```shell
   django-admin startapp users  
   ```

   其中 users 指的是你的应用名字,apps文件夹下回出现users文件夹

   ![image-20230813163312058](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20230813163312058.png)

   创建完成后，需要到与工程名相同的文件夹下（这里是ezfy）的 `settings.py` 中INSTALLED_APPS进行注册。一定要注册！

   ![image-20230813163411551](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20230813163411551.png)

   ![image-20230813164027839](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20230813164027839.png)

   * 打开 apps 下users下的 apps.py文件，修改为对应的apps.users. 
   * 将 name 变量赋值修改 

   ![image-20230813163623704](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20230813163623704.png)

   - 迁移应用	

   ![image-20230813164529447](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20230813164529447.png)

   - 运行

   ![image-20230813164658741](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20230813164658741.png)

