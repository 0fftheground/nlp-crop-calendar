#!/usr/bin/env python
"""
启动 Chainlit 应用脚本
基于 Chainlit 框架的 NLP Agent 对话应用
"""

import os
import sys
import argparse
import logging
import subprocess

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="启动 NLP Agent Chainlit 应用",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python run_chainlit.py                    # 启动默认应用
    python run_chainlit.py --port 8080        # 使用 8080 端口
    python run_chainlit.py --host 0.0.0.0     # 监听所有接口
    python run_chainlit.py --watch             # 启用热重载
        """
    )

    parser.add_argument(
        '--host',
        type=str,
        default='localhost',
        help='服务器主机地址 (默认: localhost)'
    )

    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='服务器端口 (默认: 8000)'
    )

    parser.add_argument(
        '--watch',
        dest='watch',
        action='store_true',
        default=True,
        help='启用代码变化监控（热重载），默认开启'
    )
    parser.add_argument(
        '--no-watch',
        dest='watch',
        action='store_false',
        help='关闭热重载'
    )

    parser.add_argument(
        '--headless',
        action='store_true',
        help='以无头模式运行（不打开浏览器）'
    )

    parser.add_argument(
        '--llm',
        type=str,
        choices=['openai', 'ollama', 'mock'],
        default='mock',
        help='LLM 提供商 (默认: mock)'
    )

    args = parser.parse_args()

    # 配置环境变量
    os.environ['DEFAULT_LLM_PROVIDER'] = args.llm

    # 构建 Chainlit 命令
    cmd = [
        'chainlit',
        'run',
        'chainlit_app.py',
        '--host', args.host,
        '--port', str(args.port),
    ]

    # 添加可选参数
    if args.watch:
        cmd.append('--watch')

    if args.headless:
        cmd.append('--headless')

    # 启动信息
    logger.info(f"启动 Chainlit 应用...")
    logger.info(f"主机: {args.host}")
    logger.info(f"端口: {args.port}")
    logger.info(f"LLM 提供商: {args.llm}")
    logger.info(f"热重载: {args.watch}")
    
    url = f"http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}"
    logger.info(f"访问地址: {url}")

    try:
        # 执行 Chainlit 命令
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        logger.info("应用已停止")
    except subprocess.CalledProcessError as e:
        logger.error(f"应用启动失败: {str(e)}")
        sys.exit(1)
    except FileNotFoundError:
        logger.error("Chainlit 未安装，请运行: pip install chainlit")
        sys.exit(1)


if __name__ == '__main__':
    main()
