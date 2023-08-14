# 4. 模型概念

## 4.1 使用Django进行数据库开发 

- `MVT`设计模式中的`Model`, 专门负责和数据库交互.对应`(models.py)`
- 由于`Model`中内嵌了`ORM框架`, 所以不需要直接面向数据库编程.
- 而是定义模型类, 通过`模型类和对象`完成数据库表的`增删改查`.
- `ORM框架`就是把数据库表的行与相应的对象建立关联, 互相转换.使得数据库的操作面向对象.

## 4.2 使用Django进行数据库开发的步骤 

1. 定义模型类
2. 模型迁移
3. 操作数据库(**提示：默认采用**`sqlite3`**数据库来存储数据**)

#### 4.2.1 定义模型类

- 根据书籍表结构设计模型类

- 首先在`models.py`下定义模型类,继承自`models.Model`

  - ```python
    #  产品表 
    from django.db import models
    class GoodsCategory(models.Model):
        """产品分类"""
        '''  
        - name: 分类名称，最大长度为64个字符。
        - remark: 备注，最大长度为256个字符。可以为空。
        '''
        name = CharField(max_length=64, verbose_name='名称')
        remark = CharField(max_length=256, null=True, blank=True, verbose_name='备注')
    ```

- 根据设计产品信息 表 类：

  - ```python
    #  产品信息 表 
    class Goods(models.Model):
        """产品"""
        '''
        - number: 产品编号，最大长度为32个字符。
        - name: 产品名称，最大长度为64个字符。
        - barcode: 产品条码，最大长度为32个字符。可以为空。
        - category: 产品分类，是一个外键字段，关联到GoodsCategory模型类。当Goods对象被删除时，与之关联的GoodsCategory对象的值将被设置为NULL。使用related_name='goods_set'来定义反向关联的名称。
        - spec: 产品规格，最大长度为64个字符。可以为空。
        - shelf_life_days: 保质期天数，一个整数字段。可以为空。
        - purchase_price: 采购价，一个浮点数字段，默认值为0。
        - retail_price: 零售价，一个浮点数字段，默认值为0。
        - remark: 备注，最大长度为256个字符。可以为空。
        '''
        number = CharField(max_length=32, verbose_name='编号')
        name = CharField(max_length=64, verbose_name='名称')
        barcode = CharField(max_length=32, null=True, blank=True, verbose_name='条码')
        category = ForeignKey('goods.GoodsCategory', on_delete=SET_NULL, null=True,related_name='goods_set', verbose_name='产品分类')
        spec = CharField(max_length=64, null=True, blank=True, verbose_name='规格')
        shelf_life_days = IntegerField(null=True, verbose_name='保质期天数')
        purchase_price = FloatField(default=0, verbose_name='采购价')
        retail_price = FloatField(default=0, verbose_name='零售价')
        remark = CharField(max_length=256, null=True, blank=True, verbose_name='备注')
    ```

- 说明 :

  - 产品分类-产品信息的关系为一对多. 一个产品中可以有多个分类.
  - 不需要定义主键字段, 在生成表时会自动添加, 并且值为自增长.

- 根据数据库表的设计

#### 4.2.2 模型迁移 （建表）

这两个命令是Django框架中的关键命令，用于进行数据库迁移。当你修改了Django模型后，你需要运行这两个命令，以将这些更改应用到数据库中。

1. `python manage.py makemigrations`: 这个命令用于生成迁移脚本。当你更新了模型文件之后，需要运行该命令，Django会检测模型的改变，然后自动生成相应的迁移脚本，存储在`migrations/`​目录下。通常来说，你需要针对每个应用运行一次该命令。
2. `python manage.py migrate`: 这个命令用于将迁移脚本应用到数据库中。当你在模型文件中进行更改之后，需要先通过`makemigrations`​命令生成迁移脚本，然后运行该命令将这些脚本应用到数据库中。对于新的迁移脚本，Django会逐个执行它们，从而更新数据库结构。对于已经执行过的脚本，Django会跳过它们，避免重复执行。

这两个命令是Django框架中非常重要的命令，在修改数据库相关内容时必须时刻清醒地记住使用它们。

- 迁移由两步完成 :

  - 生成迁移文件：根据模型类生成创建表的语句

    ```
    python manage.py makemigrations
    ```

    ![image-20230814220637816](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20230814220637816.png)

  - 执行迁移：根据第一步生成的语句在数据库中创建表

    ```
    python manage.py migrate
    ```

    ![image-20230814220657509](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images/image-20230814220657509.png)

