#!/usr/bin/env python
"""
启动 FastAPI Web 应用脚本
支持配置 FastAPI 应用的各种参数
"""

import os
import sys
import argparse
import logging
import uvicorn

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.web_app_fastapi import create_app

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="启动 NLP Agent FastAPI Web 应用",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python run_web.py                    # 启动默认应用
    python run_web.py --port 8080        # 使用 8080 端口
    python run_web.py --llm openai       # 使用 OpenAI
    python run_web.py --reload           # 启用热重载
        """
    )

    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='服务器主机地址 (默认: 0.0.0.0)'
    )

    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='服务器端口 (默认: 5000)'
    )

    parser.add_argument(
        '--reload',
        action='store_true',
        help='启用自动重载（开发模式）'
    )

    parser.add_argument(
        '--llm',
        type=str,
        choices=['openai', 'ollama', 'mock'],
        default='mock',
        help='LLM 提供商 (默认: mock)'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='工作进程数 (默认: 1)'
    )

    args = parser.parse_args()

    # 配置环境变量
    os.environ['DEFAULT_LLM_PROVIDER'] = args.llm

    # 创建应用
    logger.info("正在创建 FastAPI 应用...")
    app = create_app()

    # 启动应用
    logger.info(f"启动 FastAPI 服务器: http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}")
    logger.info(f"LLM 提供商: {args.llm}")
    logger.info(f"自动重载: {args.reload}")
    logger.info(f"工作进程数: {args.workers}")
    logger.info(f"API 文档: http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}/docs")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level="info"
    )


if __name__ == '__main__':
    main()
