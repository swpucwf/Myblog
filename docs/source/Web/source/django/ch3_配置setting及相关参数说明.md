# 3. 配置 settings.py 及相关参数说明

## 3.1 配置setting.py文件

1. 设置**setting.py**文件

   加入安装的库

   ```python
   'apps.erp_test',
   'rest_framework',
   'django_filters',
   'drf_spectacular',
      
   ```

   加入新增的APP

   ```python
    'users'
   ```

2. 启动项目

   ```python
   # 运行项目先执行数据库相关操作，再启动 django 项目
   python manage.py makemigrations
   python manage.py migrate
   python manage.py runserver
   ```

## 3.2 相关参数说明

#### 3.2.1 BASE_DIR

```
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
```

 当前工程的根目录，Django会依此来定位工程内的相关文件，我们也可以使用该参数来构造文件路径。

#### 3.2.2 DEBUG

```python
调试模式，创建工程后初始值为True，即默认工作在调试模式下。
作用：
修改代码文件，程序自动重启
Django程序出现异常时，向前端显示详细的错误追踪信息.而非调试模式下，仅返回Server Error (500)
```

注意：部署线上运行的Django不要运行在调式模式下，记得修改DEBUG=False和ALLOW_HOSTS。

## 3.2.3 本地语言与时区

Django支持本地化处理，即显示语言与时区支持本地化。

本地化是将显示的语言、时间等使用本地的习惯，这里的本地化就是进行中国化，中国大陆地区使用**简体中文**，时区使用**亚洲/上海**时区，注意这里不使用北京时区表示。

初始化的工程默认语言和时区为英语和UTC标准时区

```python
LANGUAGE_CODE = 'en-us'  # 语言
TIME_ZONE = 'UTC'  # 时区# 时区
```

将语言和时区修改为中国大陆信息

```
LANGUAGE_CODE = 'zh-Hans'
TIME_ZONE = 'Asia/Shanghai'
```

### 3.3 静态文件

项目中的CSS、图片、js都是静态文件。一般会将静态文件放到一个单独的目录中，以方便管理。在html页面中调用时，也需要指定静态文件的路径，Django中提供了一种解析的方式配置静态文件路径。静态文件可以放在项目根目录下，也可以放在应用的目录下，由于有些静态文件在项目中是通用的，所以推荐放在项目的根目录下，方便管理。

为了提供静态文件，需要配置两个参数：

- **STATICFILES_DIRS**存放查找静态文件的目录
- **STATIC_URL**访问静态文件的URL前缀

## 示例

1） 在项目根目录下创建static目录来保存静态文件。

2） 在ezfy/settings.py中修改静态文件的两个参数为

```
STATIC_URL = '/static/'
STATICFILES_DIRS = [
    os.path.join(BASE_DIR, 'static'),
]
```

3）此时在static添加的任何静态文件都可以使用网址**/static/文件在static中的路径**来访问了。

例如，我们向static目录中添加一个index.html文件，在浏览器中就可以使用127.0.0.1:8000/static/index.html来访问。

或者我们在static目录中添加了一个子目录和文件book/detail.html，在浏览器中就可以使用127.0.0.1:8000/static/book/detail.html来访问。

### 3.4 App应用配置

在每个应用目录中都包含了apps.py文件，用于保存该应用的相关信息。

在创建应用时，Django会向apps.py文件中写入一个该应用的配置类，如

```
from django.apps import AppConfig


class BookConfig(AppConfig):
    name = 'user'
```

我们将此类添加到工程settings.py中的INSTALLED_APPS列表中，表明注册安装具备此配置属性的应用。

- **AppConfig.name**属性表示这个配置类是加载到哪个应用的，每个配置类必须包含此属性，默认自动生成。
- **AppConfig.verbose_name**属性用于设置该应用的直观可读的名字，此名字在Django提供的Admin管理站点中会显示，如

```
from django.apps import AppConfig

class UsersConfig(AppConfig):
    name = 'user'
    verbose_name = '图书管理员'
```

