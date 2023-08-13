[TOC]

# django 环境安装

## 1. 安装环境

### 1.1 安装 Python (配置虚拟环境)

#### 1.1.1 虚拟环境安装

- **提示**：使用如上命令, 会将Django安装到`/usr/local/lib/python2.7/dist-packages`路径下

- **问题**：如果在一台电脑上, 想开发多个不同的项目, 需要用到同一个包的不同版本, 如果使用上面的命令, 在同一个目录下安装或者更新, 新版本会覆盖以前的版本, 其它的项目就无法运行了.

- 解决方案

  - **作用**:`虚拟环境`可以搭建独立的`python运行环境`, 使得单个项目的运行环境与其它项目互不影响.
  - 所有的`虚拟环境`都位于`/home/`下的隐藏目录`.virtualenvs`下

- 安装

  ```shell
  pip install virtualenv
  pip install virtualenvwrapper
  ```

- 创建虚拟环境的命令 :

  ```shell
  mkvirtualenv 虚拟环境名称
  例 ：
  mkvirtualenv django4_2
  ```

  ```
  mkvirtualenv -p python3 虚拟环境名称
  例 ：
  mkvirtualenv -p python3 django4_2
  ```

- 查看虚拟环境的命令 

  ```shell
  workon
  ```

- 使用虚拟环境的命令 

  ```shell
  workon 虚拟环境名称
  
  例 ：使用django4_2的虚拟环境
  workon django4_2
  ```

- 退出虚拟环境的命令 :

  ```
  deactivate
  ```

- 删除虚拟环境的命令 :

  ```shell
  rmvirtualenv 虚拟环境名称
  
  例 ：删除虚拟环境django4_2
  
  先退出：deactivate
  再删除：rmvirtualenv django4_2
  ```

- 如何在虚拟环境中安装工具包

  ```shell
  pip install django==4.2
  ```

- 查看虚拟环境中安装的包 :

  ```shell
  pip list
  ```

由于国外源速度慢，可以pip添加清华源

```python
 pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple 
```

#### 1.1.2 步骤

1. 创建虚拟环境

   ```shell
   python -m venv django4_2
   ```

   > erp_venv 为虚拟环境的名字

   Python 虚拟环境，并将其安装在你当前所在目录下的 `erp_venv`​​ 文件夹中。虚拟环境可以帮助你隔离不同的项目的依赖库，这样可以避免项目间的库版本冲突。

2. 启动虚拟环境

   进入虚拟环境目录 

   ```shell
   cd ./Scripts/activate
   ```

3. 退出虚拟环境

   ```shell
   deactivate
   ```

4. 安装 Django

   ```shell
   pip install django
   ```

   Django 是一个 Python web 框架，提供许多功能，如 ORM、认证、表单、模板等，它可以帮助你更快、更轻松地开发 web 应用程序。

5. 安装 DRF

   ```shell
   pip install djangorestframework
   ```

   DRF 是一个基于 Django 的强大而灵活的 RESTful 框架，它提供了许多工具和库，可帮助你快速开发基于 RESTful API 的 web 应用程序。

6. 安装 Django-Filter

   ```shell
   pip install django-filter
   ```

   介绍：[Integration with DRF — django-filter 23.2 documentation](https://django-filter.readthedocs.io/en/stable/guide/rest_framework.html#drf-integration)

   Django-Filter 是一个基于 Django 的库，它提供了一种简单、灵活的方式来过滤 Django 模型的查询集。Django-Filter 的 API 允许开发者使用简单的查询表达式，构建和应用复杂的过滤器，从而在查询集中选择和排除数据。

   Django-Filter 通过与 DRF Spectacular 的集成，支持 OpenAPI 规范表述的数据过滤和查询，提供了更加优雅的 API 规范方案。

7. 安装 Django Spectacular

   ```shell
   pip install drf_spectacular
   ```

   介绍：DRF Spectacular 是 DRF 的 OpenAPI 规范工具。它可以自动构建和生成 OpenAPI 规范文档，并提供方便的 API 测试工具，使你能够更加轻松地创建、测试和维护 RESTful API。同时，它也支持集成 Django Filter，允许你通过 URL 参数过滤查询数据。

### 1.2 Conda配置环境

1. conda 配置python解释器,环境名为**django4_2**

   ```shell
   conda create -n django4_2 python=3.8
   # 激活环境
   conda activate django4_2
   ```

   ![image-20230813135032638](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20230813135032638.png)

   ![image-20230813135129709](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20230813135129709.png)

2. 安装django==4.2库

   ```shell
   pip install django==4.2
   ```

   ![image-20230813142045532](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20230813142045532.png)

3. 安装DRF

   ```shell
   pip install djangorestframework
   ```

   ![image-20230813143340454](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20230813143340454.png)

4. 安装 Django-Filter

   ```shel
   pip install django-filter
   ```

   ![image-20230813143617191](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20230813143617191.png)

5. 在虚拟环境中，安装 debug_toolbar 库

   ```sh
   pip install django-debug-toolbar
   ```

   ![image-20230813143813760](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20230813143813760.png)

6. 在虚拟环境中，安装 django_extensions 库

   ```shell
   pip install django_extensions
   ```

   ![image-20230813143831547](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20230813143831547.png)







　　‍