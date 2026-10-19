"""数据库引擎与会话工厂：共享连接池，不共享一次事务的 Session。"""
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import URL, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine


# 这里读取数据库配置；app/config.py 的 Settings 读取模型等应用配置。
# 拆开是为了避免将数据库密码混入模型配置，也便于单测注入独立数据库。
class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(
        # 从当前文件定位项目根目录，不依赖启动终端所在的目录。
        env_file=Path(__file__).resolve().parent.parent / '.env.mysql',
        # .env.mysql 还含有容器初始化用的 root 密码；extra='ignore' 忽略它，
        # 应用只使用权限更有限的 mewhelp_app 账号。系统环境变量优先于文件。
        env_file_encoding='utf-8', extra='ignore',
    )
    # FastAPI 运行在 Windows 主机，因此连 Docker 映射到本机的端口。
    # 若将来应用也放进容器，主机地址需要重新按容器网络配置。
    mysql_host: str = '127.0.0.1'
    mysql_port: int = Field(default=3306, ge=1, le=65535)
    mysql_database: str = 'mewhelp'
    mysql_user: str = 'mewhelp_app'
    # SecretStr 仅在显示对象时遮盖内容，不是加密存储。
    mysql_password: SecretStr

    def url(self):
        """
        URL.create 正确处理密码特殊字符；不得打印真实连接串。
        mysql 是数据库类型，asyncmy 是异步驱动；utf8mb4 支持中文及 emoji。
        只有交给驱动建立连接时才取出密码明文，不把它传给浏览器或模型。
        """
        return URL.create('mysql+asyncmy', username=self.mysql_user,
                          password=self.mysql_password.get_secret_value(),
                          host=self.mysql_host, port=self.mysql_port,
                          database=self.mysql_database, query={'charset': 'utf8mb4'})


class Database:
    # 一个应用共享一个 Database/Engine；每次仓库操作另建一个 AsyncSession。
    # Engine 管连接池，Session 管本次数据库操作，两者都不是聊天会话。
    def __init__(self, url):
        """
        pre_ping 在取出池中连接时检查可用性，避免重用已被 MySQL 断开的连接。
        echo=False 不把 SQL 和参数逐条输出到日志。
        """
        self.engine = create_async_engine(url, pool_pre_ping=True, echo=False)
        # 这是“创建 Session 的工厂”，不是已经打开的共享 Session。
        # expire_on_commit=False 让提交后已加载的属性仍可读，避免访问属性
        # 时隐式再发数据库请求；不代表对象会自动同步其他事务的新数据。
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)

    async def check(self):
        """
        启动时借一条连接做最小连通性检查，退出 with 后归还连接。
        SELECT 1 不访问业务表，所以此处成功不等于四张表已正确初始化。
        """
        async with self.engine.connect() as conn:
            await conn.execute(text('SELECT 1'))

    async def close(self):
        """
        应用关闭时释放池中连接；不删除数据库、表或持久化文件。
        """
        await self.engine.dispose()
