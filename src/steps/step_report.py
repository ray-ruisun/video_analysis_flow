#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
报告生成步骤

负责将分析结果生成 Word 文档报告:
- 汇总各视频的分析结果
- 展示跨视频共识
- 可选插入截图
"""

from pathlib import Path
from typing import List
from loguru import logger

from .base import (
    PipelineStep,
    ReportInput,
    ReportOutput,
    VideoMetrics,
    ConsensusOutput
)

# 导入原有的报告生成函数
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from report_word import generate_word_report


class ReportGenerationStep(PipelineStep[ReportInput, ReportOutput]):
    """
    报告生成步骤
    
    输入: ReportInput (视频分析结果、共识、输出路径)
    输出: ReportOutput (报告文件路径)
    
    使用示例:
        step = ReportGenerationStep()
        input_data = ReportInput(
            video_metrics=[vm1, vm2, vm3],
            consensus=consensus_output,
            output_path="report.docx",
            show_screenshots=True
        )
        output = step.run(input_data)
        print(f"报告已生成: {output.report_path}")
    """
    
    @property
    def name(self) -> str:
        return "报告生成"
    
    @property
    def description(self) -> str:
        return "将分析结果生成 Word 文档报告"
    
    def run(self, input_data: ReportInput) -> ReportOutput:
        """
        执行报告生成
        
        处理流程:
        1. 转换数据格式
        2. 调用 Word 报告生成器
        3. 返回报告路径
        """
        self.log_start(input_data)
        
        try:
            # 转换 VideoMetrics 为字典格式
            video_metrics_dicts = [vm.to_dict() for vm in input_data.video_metrics]
            
            # 转换 ConsensusOutput 为字典格式
            consensus_dict = input_data.consensus.to_dict() if input_data.consensus else {}
            
            # 调用原有的报告生成函数
            report_path = generate_word_report(
                video_metrics_dicts,
                consensus_dict,
                input_data.output_path,
                show_screenshots=input_data.show_screenshots
            )
            
            output = ReportOutput(
                success=True,
                report_path=str(report_path)
            )
            
            self.log_complete(output)
            logger.info(f"  → 报告已保存: {report_path}")
            
            return output
            
        except Exception as e:
            error_output = ReportOutput(
                success=False,
                error_message=str(e)
            )
            self.log_complete(error_output)
            raise
