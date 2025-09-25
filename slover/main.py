import time, threading, json, os, gc    # 标准库
import cv2                              # 第三方库
import numpy as np
import pytesseract , pyautogui
from PIL import Image, ImageGrab        
import tkinter as tk
from tkinter import messagebox, ttk
import concurrent.futures
# 性能优化
class PerformanceConfig:
    """性能优化配置管理"""
    def __init__(self):
        self.max_cache_size = 50
        self.max_solutions = 50
        self.ocr_timeout = 30
        self.recognition_timeout = 60
        self.optimal_thread_count = min(8, os.cpu_count() or 4)
        self.memory_cleanup_interval = 100  # 每100次操作清理一次内存
        self.operation_count = 0
        # 性能统计
        self.stats = {
            'recognition_time': [],
            'solve_time': [],
            'cache_hits': 0,
            'cache_misses': 0,
            'memory_cleanups': 0
        }
    
    def should_cleanup_memory(self):
        """检查是否需要清理内存"""
        self.operation_count += 1
        return self.operation_count % self.memory_cleanup_interval == 0
    
    def cleanup_memory(self):
        """强制垃圾回收"""
        gc.collect()
        self.stats['memory_cleanups'] += 1
        print(f"内存清理完成 (第{self.stats['memory_cleanups']}次)")
    
    def record_cache_hit(self):
        """记录缓存命中"""
        self.stats['cache_hits'] += 1
    
    def record_cache_miss(self):
        """记录缓存未命中"""
        self.stats['cache_misses'] += 1
    
    def get_cache_hit_rate(self):
        """获取缓存命中率"""
        total = self.stats['cache_hits'] + self.stats['cache_misses']
        return (self.stats['cache_hits'] / total * 100) if total > 0 else 0
    
    def print_stats(self):
        """打印性能统计信息"""
        # 避免重复输出，添加标记
        if hasattr(self, '_stats_printed'):
            return
        self._stats_printed = True
        
        print("\n=== 性能统计 ===")
        print(f"缓存命中率: {self.get_cache_hit_rate():.1f}%")
        print(f"内存清理次数: {self.stats['memory_cleanups']}")
        if self.stats['recognition_time']:
            avg_recognition = sum(self.stats['recognition_time']) / len(self.stats['recognition_time'])
            print(f"平均识别时间: {avg_recognition:.2f}秒")
        if self.stats['solve_time']:
            avg_solve = sum(self.stats['solve_time']) / len(self.stats['solve_time'])
            print(f"平均求解时间: {avg_solve:.2f}秒")
        print("===============\n")
# 全局性能配置
perf_config = PerformanceConfig()
#数独求解
class SudokuSolver:
    def __init__(self):
        self.grid = [[0 for _ in range(9)] for _ in range(9)]
        self.solutions = []
        self.current_solution_index = 0
        # 性能优化：预计算约束集合
        self.row_sets = [set() for _ in range(9)]
        self.col_sets = [set() for _ in range(9)]
        self.box_sets = [set() for _ in range(9)]
        
    def _get_box_index(self, row, col):
        """获取3x3方格的索引"""
        return (row // 3) * 3 + (col // 3)
        
    def _init_constraint_sets(self, grid):
        """初始化约束集合，大幅提升is_valid检查速度"""
        # 清空所有集合
        for i in range(9):
            self.row_sets[i].clear()
            self.col_sets[i].clear()
            self.box_sets[i].clear()
        
        # 填充已有数字到约束集合
        for i in range(9):
            for j in range(9):
                if grid[i][j] != 0:
                    num = grid[i][j]
                    self.row_sets[i].add(num)
                    self.col_sets[j].add(num)
                    self.box_sets[self._get_box_index(i, j)].add(num)
        
    def is_valid(self, grid, row, col, num):
        """优化的有效性检查，使用集合操作提升速度"""
        # 使用集合查找，时间复杂度O(1)
        return (num not in self.row_sets[row] and 
                num not in self.col_sets[col] and 
                num not in self.box_sets[self._get_box_index(row, col)])
    
    def _add_number(self, row, col, num):
        """添加数字到约束集合"""
        self.row_sets[row].add(num)
        self.col_sets[col].add(num)
        self.box_sets[self._get_box_index(row, col)].add(num)
    
    def _remove_number(self, row, col, num):
        """从约束集合中移除数字"""
        self.row_sets[row].discard(num)
        self.col_sets[col].discard(num)
        self.box_sets[self._get_box_index(row, col)].discard(num)
    
    def solve_all(self, grid, progress_callback=None, max_solutions=None, stop_flag=None):
        """优化的求解算法，找到所有可能的解"""
        self.solutions = []
        self.progress_callback = progress_callback
        self.stop_flag = stop_flag  # 添加终止标志
        self.max_solutions = max_solutions if max_solutions is not None else perf_config.max_solutions
        self.total_cells = sum(1 for i in range(9) for j in range(9) if grid[i][j] == 0)
        self.filled_cells = 0
        
        # 初始化约束集合
        self._init_constraint_sets(grid)
        
        # 获取空单元格列表并按约束数量排序（启发式优化）
        empty_cells = self._get_sorted_empty_cells(grid)
        
        # 使用优化的递归求解
        self._solve_recursive_optimized(grid, empty_cells, 0)
        return self.solutions
    
    def _get_sorted_empty_cells(self, grid):
        """获取按可能值数量排序的空单元格列表（最受约束优先）"""
        empty_cells = []
        for i in range(9):
            for j in range(9):
                if grid[i][j] == 0:
                    # 计算该位置可能的数字数量
                    possible_count = 0
                    for num in range(1, 10):
                        if self.is_valid(grid, i, j, num):
                            possible_count += 1
                    empty_cells.append((i, j, possible_count))
        
        # 按可能值数量排序，最少的优先（最受约束优先启发式）
        empty_cells.sort(key=lambda x: x[2])
        return [(row, col) for row, col, _ in empty_cells]
    
    def _solve_recursive_optimized(self, grid, empty_cells, cell_index):
        """优化的递归求解，使用启发式和约束传播"""
        # 检查终止标志
        if self.stop_flag and hasattr(self.stop_flag, 'operation_stopped') and self.stop_flag.operation_stopped:
            return
        
        # 检查是否达到最大解数量限制
        if len(self.solutions) >= self.max_solutions:
            return
            
        if cell_index >= len(empty_cells):
            # 找到一个解，保存它
            solution = [row[:] for row in grid]
            self.solutions.append(solution)
            if self.progress_callback:
                result = self.progress_callback(f"找到解 {len(self.solutions)}")
                if result is False:  # 检查回调是否要求终止
                    return
            return
        
        row, col = empty_cells[cell_index]
        
        # 获取当前位置的所有可能值
        possible_nums = []
        for num in range(1, 10):
            if self.is_valid(grid, row, col, num):
                possible_nums.append(num)
        
        # 如果没有可能的值，回溯
        if not possible_nums:
            return
        
        # 尝试每个可能的数字
        for num in possible_nums:
            # 再次检查终止标志
            if self.stop_flag and hasattr(self.stop_flag, 'operation_stopped') and self.stop_flag.operation_stopped:
                break
                
            # 检查解数量限制
            if len(self.solutions) >= self.max_solutions:
                break
                
            # 放置数字并更新约束
            grid[row][col] = num
            self._add_number(row, col, num)
            
            # 递归求解下一个位置
            self._solve_recursive_optimized(grid, empty_cells, cell_index + 1)
            
            # 回溯：移除数字并更新约束
            grid[row][col] = 0
            self._remove_number(row, col, num)
    
    def print_grid(self, grid):
        """打印数独网格"""
        for i in range(9):
            if i % 3 == 0 and i != 0:
                print("------+-------+------")
            for j in range(9):
                if j % 3 == 0 and j != 0:
                    print("|", end=" ")
                print(grid[i][j] if grid[i][j] != 0 else ".", end=" ")
            print()
#地图识别
class SudokuRecognizer:
    def __init__(self):
        self.grid_coords = None
        # 添加终止标志
        self.stop_recognition = False
        # 性能优化：使用全局配置的缓存机制
        self._image_cache = {}
        self._cell_cache = {}
        self._max_cache_size = perf_config.max_cache_size
        # OCR配置缓存
        self._ocr_configs = [
            '--psm 10 -c tessedit_char_whitelist=123456789 --oem 3',
            '--psm 8 -c tessedit_char_whitelist=123456789 --oem 3',
            '--psm 7 -c tessedit_char_whitelist=123456789 --oem 3',
            '--psm 6 -c tessedit_char_whitelist=123456789 --oem 3'
        ]
        
    def capture_screen_area(self, x, y, width, height):
        """截取屏幕指定区域"""
        screenshot = ImageGrab.grab(bbox=(x, y, x + width, y + height))
        return np.array(screenshot)
    
    def preprocess_image(self, image):
        """预处理图像以便OCR识别"""
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 增强对比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # 轻微高斯模糊去噪
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # 使用OTSU阈值和自适应阈值的组合
        _, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)
        
        # 组合两种阈值结果
        combined = cv2.bitwise_and(otsu_thresh, adaptive_thresh)
        
        return combined
    
    def extract_grid_cells(self, image, line_widths=None):
        """从图像中提取9x9网格的每个单元格，排除边框线干扰
        
        Args:
            image: 输入图像
            line_widths: 线框宽度字典 {'outer': 25, 'inner': 15, 'thick': 20}
        """
        height, width = image.shape[:2]
        cell_height = height // 9
        cell_width = width // 9
        
        cells = []
        for i in range(9):
            row = []
            for j in range(9):
                # 根据单元格位置和线框宽度动态计算边距
                if line_widths:
                    # 获取线宽参数
                    outer_width = line_widths.get('outer', 25)
                    inner_width = line_widths.get('inner', 15) 
                    thick_width = line_widths.get('thick', 20)
                    
                    # 计算上边距：考虑是否为3x3分割线或外框线
                    if i == 0:  # 顶部边缘
                        margin_top = max(4, outer_width // 2 + 3)
                    elif i % 3 == 0:  # 3x3分割线
                        margin_top = max(4, thick_width // 2 + 2)
                    else:  # 普通内部线
                        margin_top = max(3, inner_width // 2 + 2)
                    
                    # 计算下边距
                    if i == 8:  # 底部边缘
                        margin_bottom = max(4, outer_width // 2 + 3)
                    elif (i + 1) % 3 == 0:  # 下方是3x3分割线
                        margin_bottom = max(4, thick_width // 2 + 2)
                    else:  # 普通内部线
                        margin_bottom = max(3, inner_width // 2 + 2)
                    
                    # 计算左边距
                    if j == 0:  # 左边缘
                        margin_left = max(4, outer_width // 2 + 3)
                    elif j % 3 == 0:  # 3x3分割线
                        margin_left = max(4, thick_width // 2 + 2)
                    else:  # 普通内部线
                        margin_left = max(3, inner_width // 2 + 2)
                    
                    # 计算右边距
                    if j == 8:  # 右边缘
                        margin_right = max(4, outer_width // 2 + 3)
                    elif (j + 1) % 3 == 0:  # 右方是3x3分割线
                        margin_right = max(4, thick_width // 2 + 2)
                    else:  # 普通内部线
                        margin_right = max(3, inner_width // 2 + 2)
                else:
                    # 默认边距（兼容旧版本）
                    margin_top = margin_bottom = max(2, int(cell_height * 0.15))
                    margin_left = margin_right = max(2, int(cell_width * 0.15))
                
                # 计算单元格边界
                y1 = i * cell_height + margin_top
                y2 = (i + 1) * cell_height - margin_bottom
                x1 = j * cell_width + margin_left
                x2 = (j + 1) * cell_width - margin_right
                
                # 确保坐标在有效范围内
                y1 = max(0, y1)
                y2 = min(height, y2)
                x1 = max(0, x1)
                x2 = min(width, x2)
                
                # 提取单元格内部区域，避免边框线
                if y2 > y1 and x2 > x1 and (y2 - y1) > 10 and (x2 - x1) > 10:
                    cell = image[y1:y2, x1:x2]
                    row.append(cell)
                else:
                    # 如果区域太小，创建空白图像
                    row.append(np.ones((10, 10), dtype=np.uint8) * 255)
            cells.append(row)
        
        return cells
    
    def visualize_cell_regions(self, image, line_widths=None):
        """可视化单元格识别区域，用于调试边距设置"""
        height, width = image.shape[:2]
        cell_height = height // 9
        cell_width = width // 9
        
        # 创建彩色图像用于可视化
        if len(image.shape) == 2:
            vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis_image = image.copy()
        
        # 为每个单元格绘制识别区域
        for i in range(9):
            for j in range(9):
                # 使用与extract_grid_cells相同的边距计算逻辑
                if line_widths:
                    outer_width = line_widths.get('outer', 25)
                    inner_width = line_widths.get('inner', 15) 
                    thick_width = line_widths.get('thick', 20)
                    
                    # 计算边距（与extract_grid_cells保持一致）
                    if i == 0:
                        margin_top = max(4, outer_width // 2 + 3)
                    elif i % 3 == 0:
                        margin_top = max(4, thick_width // 2 + 2)
                    else:
                        margin_top = max(3, inner_width // 2 + 2)
                    
                    if i == 8:
                        margin_bottom = max(4, outer_width // 2 + 3)
                    elif (i + 1) % 3 == 0:
                        margin_bottom = max(4, thick_width // 2 + 2)
                    else:
                        margin_bottom = max(3, inner_width // 2 + 2)
                    
                    if j == 0:
                        margin_left = max(4, outer_width // 2 + 3)
                    elif j % 3 == 0:
                        margin_left = max(4, thick_width // 2 + 2)
                    else:
                        margin_left = max(3, inner_width // 2 + 2)
                    
                    if j == 8:
                        margin_right = max(4, outer_width // 2 + 3)
                    elif (j + 1) % 3 == 0:
                        margin_right = max(4, thick_width // 2 + 2)
                    else:
                        margin_right = max(3, inner_width // 2 + 2)
                else:
                    margin_top = margin_bottom = max(2, int(cell_height * 0.15))
                    margin_left = margin_right = max(2, int(cell_width * 0.15))
                
                # 计算识别区域边界
                y1 = i * cell_height + margin_top
                y2 = (i + 1) * cell_height - margin_bottom
                x1 = j * cell_width + margin_left
                x2 = (j + 1) * cell_width - margin_right
                
                # 确保坐标在有效范围内
                y1 = max(0, y1)
                y2 = min(height, y2)
                x1 = max(0, x1)
                x2 = min(width, x2)
                
                # 绘制识别区域边框（绿色）
                if y2 > y1 and x2 > x1:
                    cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    # 在区域中心标注坐标
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    cv2.putText(vis_image, f'{i},{j}', (center_x-10, center_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        
        return vis_image
    
    def recognize_digit(self, cell_image, return_confidence=False):
        """识别单元格中的数字
        
        Args:
            cell_image: 单元格图像
            return_confidence: 是否返回置信度信息
            
        Returns:
            如果return_confidence=False: 返回数字(0-9)
            如果return_confidence=True: 返回(数字, 置信度字典)
        """
        # 检查图像是否为空或过小
        if cell_image is None or cell_image.size == 0:
            if return_confidence:
                return 0, {i: 0.0 for i in range(10)}
            return 0
        
        h, w = cell_image.shape[:2]
        if h < 10 or w < 10:
            if return_confidence:
                return 0, {i: 0.0 for i in range(10)}
            return 0
        
        # 快速检查是否为空白单元格
        if self.is_empty_cell(cell_image):
            if return_confidence:
                confidence = {i: 0.0 for i in range(10)}
                confidence[0] = 95.0  # 空白单元格的置信度
                return 0, confidence
            return 0
        
        # 调整图像大小以提高识别精度
        target_size = 64  # 优化尺寸平衡精度和速度
        cell_resized = cv2.resize(cell_image, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
        
        # 优化的形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        cell_resized = cv2.morphologyEx(cell_resized, cv2.MORPH_CLOSE, kernel)
        cell_resized = cv2.morphologyEx(cell_resized, cv2.MORPH_OPEN, kernel)
        
        # 针对1/4/7的特殊处理
        cell_enhanced = self.enhance_for_147(cell_resized)
        
        # 添加边框
        bordered = cv2.copyMakeBorder(cell_enhanced, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
        
        # 性能优化：使用缓存的OCR配置
        configs = self._ocr_configs
        
        # 缓存检查：为图像生成哈希键
        import hashlib
        image_hash = hashlib.md5(bordered.tobytes()).hexdigest()
        if image_hash in self._cell_cache:
            perf_config.record_cache_hit()
            cached_result = self._cell_cache[image_hash]
            if return_confidence:
                return cached_result
            return cached_result[0] if isinstance(cached_result, tuple) else cached_result
        
        # 记录缓存未命中
        perf_config.record_cache_miss()
        
        # 多种图像预处理方法
        processed_images = [bordered]
        
        # 方法1：增强对比度
        try:
            enhanced = cv2.convertScaleAbs(bordered, alpha=1.5, beta=10)
            processed_images.append(enhanced)
        except:
            pass
        
        # 方法2：轻微膨胀（针对细线条数字如1、7）
        try:
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            dilated = cv2.dilate(bordered, kernel_dilate, iterations=1)
            processed_images.append(dilated)
        except:
            pass
        
        # 方法3：轻微腐蚀（针对粗线条数字如4、8、9）
        try:
            kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            eroded = cv2.erode(bordered, kernel_erode, iterations=1)
            processed_images.append(eroded)
        except:
            pass
        
        # 收集所有识别结果
        all_results = []
        confidence_scores = {i: 0.0 for i in range(10)}
        best_confidence = 0
        
        for img in processed_images:
            for config in configs:
                try:
                    # 使用置信度数据
                    data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)
                    
                    for i, text in enumerate(data['text']):
                        if text.strip().isdigit() and 1 <= int(text.strip()) <= 9:
                            digit = int(text.strip())
                            conf = float(data['conf'][i])
                            if conf > 0:  # 只考虑有效的置信度
                                all_results.append((digit, conf))
                                confidence_scores[digit] = max(confidence_scores[digit], conf)
                                best_confidence = max(best_confidence, conf)
                    
                    # 备用：使用简单的文本识别
                    text = pytesseract.image_to_string(img, config=config).strip()
                    if text.isdigit() and 1 <= int(text) <= 9:
                        digit = int(text)
                        # 给一个基础置信度
                        base_conf = 60.0
                        all_results.append((digit, base_conf))
                        confidence_scores[digit] = max(confidence_scores[digit], base_conf)
                        best_confidence = max(best_confidence, base_conf)
                    
                    # 如果已经有很高的置信度，可以提前退出
                    if best_confidence >= 95:
                        break
                        
                except Exception as e:
                    continue
            
            # 如果已经有很高的置信度，跳过其他图像处理方法
            if best_confidence >= 95:
                break
        
        # 如果OCR有结果，分析置信度
        if all_results:
            from collections import Counter, defaultdict
            
            # 统计每个数字的出现次数和平均置信度
            digit_stats = defaultdict(list)
            for digit, conf in all_results:
                digit_stats[digit].append(conf)
            
            # 计算加权置信度（考虑出现频率和置信度）
            final_scores = {}
            for digit, confs in digit_stats.items():
                avg_conf = sum(confs) / len(confs)
                frequency_bonus = min(len(confs) * 10, 30)  # 频率奖励，最多30分
                final_scores[digit] = avg_conf + frequency_bonus
            
            # 特殊处理：如果同时识别出1/4/7，使用特征分析来决定
            confused_digits = [d for d in [1, 4, 7] if d in final_scores]
            if len(confused_digits) >= 2:
                # 只有在真正需要时才进行复杂的特征分析
                max_conf_diff = max(final_scores.values()) - min(final_scores.values())
                if max_conf_diff < 40:  # 只有置信度差距较小时才进行特征分析
                    features = self.analyze_digit_features(cell_resized)
                else:
                    features = None
                    
                if features:
                    # 提取特征
                    top_density = features['top_horizontal_density']
                    center_vertical = features['center_vertical_density']
                    width_var = features['width_variance']
                    symmetry = features['symmetry']
                    middle_density = features['middle_horizontal_density']
                    left_vertical = features['left_vertical_density']
                    right_vertical = features['right_vertical_density']
                    corner_count = features['corner_count']
                    density = np.sum(cell_resized == 0) / cell_resized.size
                    
                    # 计算1的特征匹配度
                    one_score = 0
                    if center_vertical > top_density * 2.5:
                        one_score += 35
                    if width_var < 40:
                        one_score += 25
                    if symmetry > 0.75:
                        one_score += 25
                    if density < 0.15:
                        one_score += 15
                    if corner_count <= 2:
                        one_score += 10
                    
                    # 计算4的特征匹配度
                    four_score = 0
                    if middle_density > top_density * 1.2:
                        four_score += 30
                    if left_vertical > 0 and right_vertical > 0:
                        four_score += 25
                    if corner_count >= 3:
                        four_score += 20
                    if 0.15 <= density <= 0.30:
                        four_score += 15
                    
                    # 计算7的特征匹配度
                    seven_score = 0
                    if top_density > center_vertical * 1.0:
                        seven_score += 35
                    if right_vertical > left_vertical * 1.5:
                        seven_score += 25
                    if width_var > 25:
                        seven_score += 20
                    if 0.12 <= density <= 0.25:
                        seven_score += 15
                    
                    # 根据特征匹配度调整置信度
                    scores = {'1': one_score, '4': four_score, '7': seven_score}
                    max_feature_score = max(scores.values())
                    best_feature_digit = [k for k, v in scores.items() if v == max_feature_score][0]
                    
                    if max_feature_score >= 60:  # 特征匹配度足够高
                        best_digit = int(best_feature_digit)
                        if best_digit in final_scores:
                            final_scores[best_digit] += 30  # 大幅增强最匹配数字的置信度
                            
                            # 降低其他混淆数字的置信度
                            for digit in confused_digits:
                                if digit != best_digit:
                                    final_scores[digit] -= 20
                            
                            print(f"特征分析：更像数字{best_digit} (1:{one_score}, 4:{four_score}, 7:{seven_score})")
                    else:
                        print(f"特征分析：置信度不足 (1:{one_score}, 4:{four_score}, 7:{seven_score})")
            
            # 更新置信度分数
            for digit, score in final_scores.items():
                confidence_scores[digit] = min(max(score, 0), 99.0)  # 限制置信度范围
            
            # 选择置信度最高的数字
            best_digit = max(final_scores.keys(), key=lambda x: final_scores[x])
            
            if return_confidence:
                return best_digit, confidence_scores
            return best_digit
        
        # 如果OCR失败，使用基于特征的识别
        feature_result = self.recognize_by_features(cell_image)
        if feature_result != 0:
            confidence_scores[feature_result] = 45.0  # 特征识别的基础置信度
        
        # 性能优化：缓存结果
        result = (feature_result, confidence_scores) if return_confidence else feature_result
        self._cache_result(image_hash, result)
        
        # 内存管理：定期清理内存
        if perf_config.should_cleanup_memory():
            perf_config.cleanup_memory()
        
        if return_confidence:
            return feature_result, confidence_scores
        return feature_result
    
    def _cache_result(self, image_hash, result):
        """缓存识别结果，管理缓存大小"""
        # 如果缓存已满，删除最旧的条目
        if len(self._cell_cache) >= self._max_cache_size:
            # 删除第一个（最旧的）条目
            oldest_key = next(iter(self._cell_cache))
            del self._cell_cache[oldest_key]
        
        self._cell_cache[image_hash] = result
    
    def is_empty_cell(self, cell_image):
        """快速检查是否为空白单元格"""
        try:
            # 计算黑色像素比例
            black_pixels = np.sum(cell_image == 0)
            total_pixels = cell_image.size
            density = black_pixels / total_pixels
            return density < 0.03  # 黑色像素太少认为是空格
        except:
            return True
    
    def enhance_for_147(self, cell_image):
        """针对1/4/7数字的特殊增强处理"""
        try:
            # 轻微锐化以增强边缘
            kernel_sharpen = np.array([[-1,-1,-1],
                                     [-1, 9,-1],
                                     [-1,-1,-1]])
            sharpened = cv2.filter2D(cell_image, -1, kernel_sharpen)
            
            # 确保值在有效范围内
            sharpened = np.clip(sharpened, 0, 255)
            
            # 轻微膨胀以加粗线条
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            enhanced = cv2.dilate(sharpened, kernel, iterations=1)
            
            return enhanced.astype(np.uint8)
        except:
            return cell_image
    
    def analyze_digit_features(self, cell_image):
        """分析数字的详细特征，用于区分1/4/7"""
        try:
            h, w = cell_image.shape
            
            # 计算水平投影（每行的黑色像素数）
            horizontal_projection = np.sum(cell_image == 0, axis=1)
            
            # 计算垂直投影（每列的黑色像素数）
            vertical_projection = np.sum(cell_image == 0, axis=0)
            
            # 特征1：顶部水平线检测（7的特征）
            top_quarter = h // 4
            top_horizontal_density = np.mean(horizontal_projection[:top_quarter])
            
            # 特征2：垂直线检测（1的特征）
            center_col_start = w // 3
            center_col_end = 2 * w // 3
            center_vertical_density = np.mean(vertical_projection[center_col_start:center_col_end])
            
            # 特征3：宽度分布
            non_zero_rows = horizontal_projection > 0
            if np.any(non_zero_rows):
                width_variance = np.var(horizontal_projection[non_zero_rows])
            else:
                width_variance = 0
            
            # 特征4：左右对称性
            left_half = vertical_projection[:w//2]
            right_half = vertical_projection[w//2:]
            if len(left_half) == len(right_half):
                symmetry = 1 - np.mean(np.abs(left_half - right_half)) / (np.mean(left_half) + 1)
            else:
                symmetry = 0
            
            # 特征5：中间水平线检测（4的特征）
            middle_start = h // 3
            middle_end = 2 * h // 3
            middle_horizontal_density = np.mean(horizontal_projection[middle_start:middle_end])
            
            # 特征6：左侧垂直线检测（4的特征）
            left_quarter = w // 4
            left_vertical_density = np.mean(vertical_projection[:left_quarter])
            
            # 特征7：右侧垂直线检测（4的特征）
            right_quarter = 3 * w // 4
            right_vertical_density = np.mean(vertical_projection[right_quarter:])
            
            # 特征8：底部水平线检测
            bottom_quarter = 3 * h // 4
            bottom_horizontal_density = np.mean(horizontal_projection[bottom_quarter:])
            
            # 特征9：连通性分析
            _, labels, stats, _ = cv2.connectedComponentsWithStats(255 - cell_image, connectivity=8)
            num_components = len(stats) - 1
            
            # 特征10：角点检测（4的特征）
            corners = cv2.goodFeaturesToTrack(cell_image, maxCorners=10, qualityLevel=0.01, minDistance=5)
            corner_count = len(corners) if corners is not None else 0
            
            return {
                'top_horizontal_density': top_horizontal_density,
                'center_vertical_density': center_vertical_density,
                'width_variance': width_variance,
                'symmetry': symmetry,
                'middle_horizontal_density': middle_horizontal_density,
                'left_vertical_density': left_vertical_density,
                'right_vertical_density': right_vertical_density,
                'bottom_horizontal_density': bottom_horizontal_density,
                'num_components': num_components,
                'corner_count': corner_count,
                'horizontal_projection': horizontal_projection,
                'vertical_projection': vertical_projection
            }
        except:
            return None
    
    def recognize_by_features(self, cell_image):
        """基于特征的数字识别（备用方法）"""
        try:
            # 计算基本特征
            black_pixels = np.sum(cell_image == 0)
            total_pixels = cell_image.size
            density = black_pixels / total_pixels
            
            # 如果黑色像素太少，可能是空格
            if density < 0.05:
                return 0
            
            # 获取详细特征分析
            features = self.analyze_digit_features(cell_image)
            
            if features:
                # 提取所有特征
                top_density = features['top_horizontal_density']
                center_vertical = features['center_vertical_density']
                width_var = features['width_variance']
                symmetry = features['symmetry']
                middle_density = features['middle_horizontal_density']
                left_vertical = features['left_vertical_density']
                right_vertical = features['right_vertical_density']
                bottom_density = features['bottom_horizontal_density']
                num_components = features['num_components']
                corner_count = features['corner_count']
                
                # 数字1的特征：主要是垂直线，对称性高，宽度变化小
                one_score = 0
                if center_vertical > top_density * 2.5:  # 垂直线明显
                    one_score += 35
                if width_var < 40:  # 宽度变化小
                    one_score += 25
                if symmetry > 0.75:  # 高对称性
                    one_score += 25
                if density < 0.15:  # 密度低
                    one_score += 15
                if corner_count <= 2:  # 角点少
                    one_score += 10
                
                # 数字4的特征：有明显的中间水平线，左右都有垂直线，角点多
                four_score = 0
                if middle_density > top_density * 1.2:  # 中间水平线明显
                    four_score += 30
                if left_vertical > 0 and right_vertical > 0:  # 左右都有垂直线
                    four_score += 25
                if corner_count >= 3:  # 角点多（4有很多转折）
                    four_score += 20
                if num_components >= 2:  # 可能有多个连通区域
                    four_score += 15
                if 0.15 <= density <= 0.30:  # 中等密度
                    four_score += 10
                
                # 数字7的特征：顶部有明显水平线，主要在右侧有垂直线
                seven_score = 0
                if top_density > center_vertical * 1.0:  # 顶部水平线明显
                    seven_score += 35
                if right_vertical > left_vertical * 1.5:  # 右侧垂直线更明显
                    seven_score += 25
                if width_var > 25:  # 宽度变化较大
                    seven_score += 20
                if 0.12 <= density <= 0.25:  # 中等偏低密度
                    seven_score += 15
                if corner_count <= 3:  # 角点适中
                    seven_score += 5
                
                # 选择得分最高的数字
                max_score = max(one_score, four_score, seven_score)
                if max_score >= 60:  # 置信度阈值
                    if max_score == one_score:
                        return 1
                    elif max_score == four_score:
                        return 4
                    elif max_score == seven_score:
                        return 7
            
            # 基于密度和连通组件的启发式判断（原有逻辑作为备用）
            if density < 0.12:
                return 1  # 1通常密度最低
            elif density > 0.25 and features and features['num_components'] >= 2:
                return 4  # 4通常密度高且有多个连通区域
            elif 0.12 <= density <= 0.20:
                return 7  # 7的密度中等
            elif density > 0.20:
                # 进一步区分其他数字
                if features and features['num_components'] == 1:
                    return 6  # 6通常是一个连通区域且密度较高
                else:
                    return 8  # 8通常有两个连通区域
            
        except:
            pass
        
        return 0
    
    def recognize_sudoku(self, image, progress_callback=None, line_widths=None, detailed_callback=None):
        """识别完整的数独网格（支持多线程和进度回调）
        
        Args:
            image: 输入图像
            progress_callback: 进度回调函数(progress_percent)
            line_widths: 线框宽度设置
            detailed_callback: 详细信息回调函数(row, col, digit, confidence_dict)
        """
        processed_image = self.preprocess_image(image)
        cells = self.extract_grid_cells(processed_image, line_widths)
        
        # 添加终止标志检查
        self.stop_recognition = False
        if stop_flag:
            self.stop_flag = stop_flag
        
        # 使用多线程加速识别
        import concurrent.futures
        import threading
        
        grid = [[0 for _ in range(9)] for _ in range(9)]
        total_cells = 81
        completed_cells = 0
        lock = threading.Lock()
        
        def recognize_cell(i, j):
            nonlocal completed_cells
            digit, confidence_dict = self.recognize_digit(cells[i][j], return_confidence=True)
            grid[i][j] = digit
            
            with lock:
                completed_cells += 1
                
                # 调用详细信息回调
                if detailed_callback:
                    detailed_callback(i, j, digit, confidence_dict)
                
                # 调用进度回调
                if progress_callback:
                    progress = (completed_cells / total_cells) * 100
                    progress_callback(progress)
        
        # 创建线程池进行并行识别
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for i in range(9):
                for j in range(9):
                    future = executor.submit(recognize_cell, i, j)
                    futures.append(future)
            
            # 等待所有任务完成
            concurrent.futures.wait(futures)
        
        return grid
    
    def recognize_sudoku_with_threads(self, image, thread_count=4, progress_callback=None, line_widths=None, detailed_callback=None, stop_flag=None):
        """优化的多线程数独识别，提升并行处理效率
        
        Args:
            image: 输入图像
            thread_count: 线程数
            progress_callback: 进度回调函数(progress_percent)
            line_widths: 线框宽度设置
            detailed_callback: 详细信息回调函数(row, col, digit, confidence_dict)
            stop_flag: 终止标志对象
        """
        processed_image = self.preprocess_image(image)
        cells = self.extract_grid_cells(processed_image, line_widths)
        
        # 使用多线程加速识别
        import concurrent.futures
        import threading
        
        grid = [[0 for _ in range(9)] for _ in range(9)]
        total_cells = 81
        completed_cells = 0
        lock = threading.Lock()
        
        # 性能优化：预先过滤空白单元格
        non_empty_tasks = []
        for i in range(9):
            for j in range(9):
                if not self.is_empty_cell(cells[i][j]):
                    non_empty_tasks.append((i, j))
                else:
                    grid[i][j] = 0
                    if detailed_callback:
                        detailed_callback(i, j, 0, {k: 0.0 for k in range(10)})
        
        def recognize_cell(task):
            nonlocal completed_cells
            i, j = task
            try:
                # 检查终止标志
                if self.stop_recognition or (hasattr(self, 'stop_flag') and hasattr(self.stop_flag, 'operation_stopped') and self.stop_flag.operation_stopped):
                    return False
                
                digit, confidence_dict = self.recognize_digit(cells[i][j], return_confidence=True)
                grid[i][j] = digit
                
                with lock:
                    completed_cells += 1
                    
                    # 调用详细信息回调
                    if detailed_callback:
                        result = detailed_callback(i, j, digit, confidence_dict)
                        if result is False:  # 如果回调返回False，表示需要终止
                            self.stop_recognition = True
                            return False
                    
                    # 调用进度回调
                    if progress_callback:
                        progress = (completed_cells / total_cells) * 100
                        result = progress_callback(progress)
                        if result is False:  # 如果回调返回False，表示需要终止
                            self.stop_recognition = True
                            return False
            except Exception as e:
                print(f"识别单元格({i},{j})时出错: {e}")
                grid[i][j] = 0
                return False
        
        # 性能优化：使用全局配置的最优线程数
        optimal_thread_count = min(thread_count, len(non_empty_tasks), perf_config.optimal_thread_count)
        
        # 创建线程池进行并行识别
        with concurrent.futures.ThreadPoolExecutor(max_workers=optimal_thread_count) as executor:
            # 批量提交任务
            futures = [executor.submit(recognize_cell, task) for task in non_empty_tasks]
            
            # 等待所有任务完成，使用配置的超时时间
            try:
                # 检查是否有任务返回False（表示需要终止）
                for future in concurrent.futures.as_completed(futures, timeout=perf_config.recognition_timeout):
                    try:
                        result = future.result()
                        if result is False or self.stop_recognition or (hasattr(self, 'stop_flag') and hasattr(self.stop_flag, 'operation_stopped') and self.stop_flag.operation_stopped):
                            # 取消所有未完成的任务
                            for f in futures:
                                f.cancel()
                            print("识别被用户终止")
                            return None  # 返回None表示被终止
                    except Exception as e:
                        print(f"任务执行出错: {e}")
            except concurrent.futures.TimeoutError:
                print(f"识别超时({perf_config.recognition_timeout}秒)，部分单元格可能未完成")
        
        return grid
#键鼠填充
class SudokuAutoFiller:
    def __init__(self):
        self.cell_positions = []
        self.status_callback = None  # GUI状态更新回调函数
        self.stop_filling = False  # 终止填充标志
        
    def calculate_cell_positions(self, start_x, start_y, cell_width, cell_height):
        """计算每个单元格的中心位置"""
        positions = []
        for i in range(9):
            row = []
            for j in range(9):
                center_x = start_x + j * cell_width + cell_width // 2
                center_y = start_y + i * cell_height + cell_height // 2
                row.append((center_x, center_y))
            positions.append(row)
        return positions
    
    def fill_cell(self, row, col, value, positions):
        """在指定位置填入数字"""
        if 0 <= row < 9 and 0 <= col < 9 and positions:
            try:
                x, y = positions[row][col]
                
                # 更新GUI进度信息和终端输出
                status_msg = f"鼠标位置X:{x} Y:{y} 填入{value} (第{row+1}行{col+1}列)"
                print(status_msg)
                if self.status_callback:
                    self.status_callback(status_msg)
                
                # 快速移动到目标位置并点击
                pyautogui.moveTo(x, y, duration=0.005)  # 快速移动到目标位置
                pyautogui.click()  # 立即点击
                time.sleep(0.001)  # 减少等待时间
                
                # 清除当前内容（如果有）
                pyautogui.hotkey('ctrl', 'a')  # 全选
                time.sleep(0.001)
                pyautogui.press('delete')
                time.sleep(0.001)
                
                # 输入数字
                if value != 0:
                    pyautogui.typewrite(str(value))
                    time.sleep(0.001)
                    # 按回车确认输入
                    pyautogui.press('enter')
                    time.sleep(0.001)
                    
            except Exception as e:
                error_msg = f"填充单元格 ({row+1}, {col+1}) 失败: {str(e)}"
                print(error_msg)
                if self.status_callback:
                    self.status_callback(error_msg)
                raise e
    
    def fill_solution(self, original_grid, solution_grid, positions, delay=0.005):
        """填充完整的解"""
        self.stop_filling = False  # 重置终止标志
        total_cells = sum(1 for i in range(9) for j in range(9) if original_grid[i][j] == 0)
        filled_cells = 0
        
        for i in range(9):
            if self.stop_filling:  # 检查是否需要终止
                if self.status_callback:
                    self.status_callback("填充已终止")
                break
            
            # 检查ESC键是否被按下
            try:
                import keyboard
                if keyboard.is_pressed('esc'):
                    self.stop_filling = True
                    if self.status_callback:
                        self.status_callback("用户按ESC键终止填充")
                    print("用户按ESC键终止填充")
                    break
            except ImportError:
                # 如果没有keyboard库，跳过ESC检测
                pass
                
            for j in range(9):
                if self.stop_filling:  # 检查是否需要终止
                    if self.status_callback:
                        self.status_callback("填充已终止")
                    break
                
                # 再次检查ESC键
                try:
                    import keyboard
                    if keyboard.is_pressed('esc'):
                        self.stop_filling = True
                        if self.status_callback:
                            self.status_callback("用户按ESC键终止填充")
                        print("用户按ESC键终止填充")
                        break
                except ImportError:
                    # 如果没有keyboard库，跳过ESC检测
                    pass
                    
                if original_grid[i][j] == 0:  # 只填充原本为空的格子
                    self.fill_cell(i, j, solution_grid[i][j], positions)
                    filled_cells += 1
                    
                    # 更新进度信息
                    progress = (filled_cells / total_cells) * 100
                    if self.status_callback:
                        self.status_callback(f"填充进度: {progress:.1f}% ({filled_cells}/{total_cells})")
                    
                    time.sleep(delay)
#线框排除
class AreaSelector:
    def __init__(self, callback):
        self.callback = callback
        self.selecting = False
        self.start_x = 0
        self.start_y = 0
        self.end_x = 0
        self.end_y = 0
        self.selection_window = None
        # 网格线配置
        self.outer_line_width = 25  # 外框线宽
        self.inner_line_width = 15  # 内部网格线宽
        self.thick_line_width = 20  # 3x3分割线宽
        
    def start_selection(self):
        """开始区域选择"""
        try:
            # 获取屏幕尺寸
            import tkinter as tk
            root = tk.Tk()
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            root.destroy()
            
            # 创建全屏透明窗口
            self.selection_window = tk.Toplevel()
            self.selection_window.geometry(f"{screen_width}x{screen_height}+0+0")
            self.selection_window.overrideredirect(True)  # 移除窗口装饰
            self.selection_window.attributes('-alpha', 0.3)
            self.selection_window.configure(bg='black')
            self.selection_window.attributes('-topmost', True)
            
            # 创建画布
            self.canvas = tk.Canvas(self.selection_window, highlightthickness=0, bg='black')
            self.canvas.pack(fill=tk.BOTH, expand=True)
            
            # 绑定鼠标事件
            self.canvas.bind('<Button-1>', self.on_click)
            self.canvas.bind('<B1-Motion>', self.on_drag)
            self.canvas.bind('<ButtonRelease-1>', self.on_release)
            self.canvas.bind('<Escape>', self.on_escape)
            
            # 绑定键盘事件
            self.selection_window.bind('<Escape>', self.on_escape)
            self.selection_window.bind('<Key>', self.on_escape)
            
            # 添加提示文字
            self.canvas.create_text(screen_width//2, 50, text="拖拽选择数独区域，松开鼠标确认选择\n按ESC键取消", 
                                   fill='white', font=('Arial', 16), justify='center')
            
            # 设置焦点
            self.selection_window.focus_set()
            self.canvas.focus_set()
            
        except Exception as e:
            if self.selection_window:
                self.selection_window.destroy()
                self.selection_window = None
            raise e
        
    def on_click(self, event):
        self.selecting = True
        self.start_x = event.x
        self.start_y = event.y
        
    def on_drag(self, event):
        if self.selecting:
            # 清除之前的选择图形
            self.canvas.delete('selection')
            
            # 计算选择区域
            x1 = min(self.start_x, event.x)
            y1 = min(self.start_y, event.y)
            x2 = max(self.start_x, event.x)
            y2 = max(self.start_y, event.y)
            
            # 绘制外框
            self.canvas.create_rectangle(x1, y1, x2, y2,
                                        outline='red', width=self.outer_line_width, tags='selection')
            
            # 绘制9x9网格线
            self.draw_grid_lines(x1, y1, x2, y2)
    
    def draw_grid_lines(self, x1, y1, x2, y2):
        """绘制9x9数独网格线"""
        width = x2 - x1
        height = y2 - y1
        
        if width < 50 or height < 50:  # 区域太小不绘制网格
            return
        
        # 计算每个单元格的大小
        cell_width = width / 9
        cell_height = height / 9
        
        # 绘制垂直线
        for i in range(1, 9):
            x = x1 + i * cell_width
            # 3x3分割线使用较粗的线
            line_width = self.thick_line_width if i % 3 == 0 else self.inner_line_width
            color = 'orange' if i % 3 == 0 else 'yellow'
            self.canvas.create_line(x, y1, x, y2, 
                                  fill=color, width=line_width, tags='selection')
        
        # 绘制水平线
        for i in range(1, 9):
            y = y1 + i * cell_height
            # 3x3分割线使用较粗的线
            line_width = self.thick_line_width if i % 3 == 0 else self.inner_line_width
            color = 'orange' if i % 3 == 0 else 'yellow'
            self.canvas.create_line(x1, y, x2, y, 
                                  fill=color, width=line_width, tags='selection')
    
    def on_release(self, event):
        if self.selecting:
            self.selecting = False
            self.end_x = event.x
            self.end_y = event.y
            
            # 确保坐标正确
            x1 = min(self.start_x, self.end_x)
            y1 = min(self.start_y, self.end_y)
            x2 = max(self.start_x, self.end_x)
            y2 = max(self.start_y, self.end_y)
            
            width = x2 - x1
            height = y2 - y1
            
            # 关闭选择窗口
            self.close_selection_window()
            
            # 回调函数传递选择的区域
            if width > 50 and height > 50:  # 最小区域限制
                self.callback(x1, y1, width, height)
    
    def on_escape(self, event):
        """ESC键取消选择"""
        self.close_selection_window()
    
    def close_selection_window(self):
        """安全关闭选择窗口"""
        try:
            if self.selection_window:
                self.selection_window.destroy()
                self.selection_window = None
        except:
            pass
#GUI界面
class SudokuGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("数独自动求解器-V4")
        window_width = 500
        window_height = 750
        screen_width = self.root.winfo_screenwidth()  # 屏幕宽度
        x_pos = screen_width - window_width  # 右上角X坐标
        y_pos = 0  # 右上角Y坐标（顶部对齐）
        self.root.geometry(f"{window_width}x{window_height}+{x_pos}+{y_pos}")

        self.root.resizable(True, True)
        self.root.attributes('-topmost', True)
        
        # 添加操作终止标志
        self.operation_stopped = False
        
        # 绑定ESC键全局事件
        self.root.bind('<Escape>', self.on_escape_key)
        self.root.bind('<KeyPress-Escape>', self.on_escape_key)
        
        self.solver = SudokuSolver()
        self.recognizer = SudokuRecognizer()
        self.auto_filler = SudokuAutoFiller()
        self.area_selector = AreaSelector(self.on_area_selected)
        
        # 设置auto_filler的状态回调函数
        self.auto_filler.status_callback = self.update_status_safe
        
        self.original_grid = None
        self.solutions = []
        self.current_solution = 0
        self.cell_positions = None
        # 设置配置文件路径为代码同目录下的sudoku_config.json
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_file = os.path.join(script_dir, "sudoku_config.json")
        self.settings_expanded = False  # 初始化设置区域展开状态
        
        self.load_config()
        self.setup_ui()
        self.apply_config_to_ui()
        
    def load_config(self):
        """加载配置文件"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    self.x_var = tk.StringVar(value=str(config.get('x', 100)))
                    self.y_var = tk.StringVar(value=str(config.get('y', 100)))
                    self.width_var = tk.StringVar(value=str(config.get('width', 450)))
                    self.height_var = tk.StringVar(value=str(config.get('height', 450)))
                    # 加载新的配置项
                    self.thread_count_value = config.get('thread_count', 4)
                    self.outer_line_width_value = config.get('outer_line_width', 25)
                    self.inner_line_width_value = config.get('inner_line_width', 15)
                    self.thick_line_width_value = config.get('thick_line_width', 20)
            else:
                self.x_var = tk.StringVar(value="100")
                self.y_var = tk.StringVar(value="100")
                self.width_var = tk.StringVar(value="450")
                self.height_var = tk.StringVar(value="450")
                # 默认值
                self.thread_count_value = 4
                self.outer_line_width_value = 25
                self.inner_line_width_value = 15
                self.thick_line_width_value = 20
        except:
            self.x_var = tk.StringVar(value="100")
            self.y_var = tk.StringVar(value="100")
            self.width_var = tk.StringVar(value="450")
            self.height_var = tk.StringVar(value="450")
            # 默认值
            self.thread_count_value = 4
            self.outer_line_width_value = 25
            self.inner_line_width_value = 15
            self.thick_line_width_value = 20
    
    def save_config(self):
        """保存配置文件"""
        try:
            config = {
                'x': int(self.x_var.get()),
                'y': int(self.y_var.get()),
                'width': int(self.width_var.get()),
                'height': int(self.height_var.get()),
                'thread_count': self.thread_count.get() if hasattr(self, 'thread_count') else self.thread_count_value,
                'outer_line_width': self.outer_line_width.get() if hasattr(self, 'outer_line_width') else self.outer_line_width_value,
                'inner_line_width': self.inner_line_width.get() if hasattr(self, 'inner_line_width') else self.inner_line_width_value,
                'thick_line_width': self.thick_line_width.get() if hasattr(self, 'thick_line_width') else self.thick_line_width_value
            }
            with open(self.config_file, 'w') as f:
                json.dump(config, f)
            print("配置已保存")
        except Exception as e:
            print(f"保存配置失败: {e}")
    
    def apply_config_to_ui(self):
        """应用配置到UI组件"""
        # 更新线程数显示
        if hasattr(self, 'thread_label'):
            self.thread_label.config(text=str(self.thread_count.get()))
        
        # 更新区域选择器的线宽设置
        if hasattr(self, 'area_selector'):
            self.area_selector.outer_line_width = self.outer_line_width.get()
            self.area_selector.inner_line_width = self.inner_line_width.get()
            self.area_selector.thick_line_width = self.thick_line_width.get()
    
    def update_status_safe(self, message):
        """线程安全地更新状态信息"""
        def update():
            self.status_var.set(message)
            self.root.update_idletasks()
        self.root.after(0, update)
    
    def update_status(self, message):
        """直接更新状态信息（非线程安全）"""
        self.status_var.set(message)
    
    def stop_current_operation(self):
        """终止当前的识别/填充/求解操作"""
        try:
            # 设置全局终止标志
            self.operation_stopped = True
            
            # 终止自动填充线程
            if hasattr(self, 'auto_filler') and self.auto_filler:
                self.auto_filler.stop_filling = True
                print("已发送终止信号给自动填充线程")
            
            # 终止识别线程
            if hasattr(self, 'recognizer') and self.recognizer:
                self.recognizer.stop_recognition = True
                print("已发送终止信号给识别线程")
            
            # 终止求解线程（通过solver的终止标志）
            if hasattr(self, 'solver') and self.solver:
                print("已发送终止信号给求解线程")
            
            # 重置进度条
            self.progress_var.set(0)
            
            # 更新状态
            self.update_status("操作已终止")
            
            print("用户终止了当前操作 - 所有线程将在下次检查时停止")
            
            # 延迟重置终止标志，为下次操作做准备
            self.root.after(2000, lambda: setattr(self, 'operation_stopped', False))
            
        except Exception as e:
            print(f"终止操作时出错: {str(e)}")
    
    def on_escape_key(self, event=None):
        """ESC键处理函数"""
        print("检测到ESC键按下")
        self.stop_current_operation()
        return 'break'
    
    def on_area_selected(self, x, y, width, height):
        """区域选择回调函数"""
        self._area_selected_recently = True
        self.x_var.set(str(x))
        self.y_var.set(str(y))
        self.width_var.set(str(width))
        self.height_var.set(str(height))
        self.save_config()
        self.update_status(f"已选择区域: ({x}, {y}) 大小: {width}x{height}")
        # 清除标记
        self.root.after(1000, lambda: setattr(self, '_area_selected_recently', False))
    
    def create_sudoku_grid(self, parent):
        """创建可编辑的数独网格"""
        self.grid_entries = []
        self.grid_frame = tk.Frame(parent)
        self.grid_frame.pack(pady=10)
        
        for i in range(9):
            row_entries = []
            for j in range(9):
                # 创建输入框
                entry = tk.Entry(self.grid_frame, width=3, justify='center', 
                               font=('Arial', 12, 'bold'))
                entry.grid(row=i, column=j, padx=1, pady=1)
                
                # 绑定验证函数
                vcmd = (self.root.register(self.validate_input), '%P')
                entry.config(validate='key', validatecommand=vcmd)
                
                # 绑定输入变化事件，用于冲突检查
                entry.bind('<KeyRelease>', lambda e: self.root.after(100, self.check_conflicts))
                entry.bind('<FocusOut>', lambda e: self.check_conflicts())
                
                # 设置九宫格边框样式
                if i % 3 == 0 and i != 0:
                    entry.grid(pady=(3, 1))
                if j % 3 == 0 and j != 0:
                    entry.grid(padx=(3, 1))
                if i % 3 == 0 and i != 0 and j % 3 == 0 and j != 0:
                    entry.grid(padx=(3, 1), pady=(3, 1))
                
                row_entries.append(entry)
            self.grid_entries.append(row_entries)
        

    
    def validate_input(self, value):
        """验证输入只能是1-9的数字或空"""
        if value == "":
            return True
        if len(value) == 1 and value.isdigit() and 1 <= int(value) <= 9:
            return True
        return False
    
    def check_conflicts(self):
        """检查编辑器中的数字冲突并标红"""
        # 获取当前网格状态
        current_grid = []
        for i in range(9):
            row = []
            for j in range(9):
                value = self.grid_entries[i][j].get().strip()
                if value and value.isdigit():
                    row.append(int(value))
                else:
                    row.append(0)
            current_grid.append(row)
        
        # 重置所有单元格颜色
        for i in range(9):
            for j in range(9):
                entry = self.grid_entries[i][j]
                if current_grid[i][j] != 0:
                    # 检查是否是原始识别的数字还是求解的数字
                    if hasattr(self, 'original_grid') and self.original_grid and self.original_grid[i][j] != 0:
                        entry.config(fg='black')  # 识别的数字保持黑色
                    else:
                        entry.config(fg='green')  # 求解的数字设为绿色
                else:
                    entry.config(fg='black')  # 空格设为黑色
        
        # 检查冲突并标红
        for i in range(9):
            for j in range(9):
                entry = self.grid_entries[i][j]
                if current_grid[i][j] != 0:
                    num = current_grid[i][j]
                    has_conflict = False
                    
                    # 检查行冲突
                    for k in range(9):
                        if k != j and current_grid[i][k] == num:
                            has_conflict = True
                            break
                    
                    # 检查列冲突
                    if not has_conflict:
                        for k in range(9):
                            if k != i and current_grid[k][j] == num:
                                has_conflict = True
                                break
                    
                    # 检查3x3方格冲突
                    if not has_conflict:
                        box_row = (i // 3) * 3
                        box_col = (j // 3) * 3
                        for r in range(box_row, box_row + 3):
                            for c in range(box_col, box_col + 3):
                                if (r != i or c != j) and current_grid[r][c] == num:
                                    has_conflict = True
                                    break
                            if has_conflict:
                                break
                    
                    # 如果有冲突，标红
                    if has_conflict:
                        entry.config(fg='red')
    
    def clear_grid(self):
        """清空数独网格"""
        for i in range(9):
            for j in range(9):
                entry = self.grid_entries[i][j]
                entry.delete(0, tk.END)
                entry.config(fg='black')  # 重置颜色为黑色
    
    def fill_from_recognition(self):
        """从识别结果填充网格"""
        if self.original_grid:
            for i in range(9):
                for j in range(9):
                    entry = self.grid_entries[i][j]
                    entry.delete(0, tk.END)
                    if self.original_grid[i][j] != 0:
                        entry.insert(0, str(self.original_grid[i][j]))
                        entry.config(fg='black')  # 识别的数字设为黑色
                    else:
                        entry.config(fg='black')  # 空格也设为黑色
            # 检查冲突并标红
            self.check_conflicts()
    
    def get_current_grid(self):
        """获取当前网格状态"""
        grid = []
        for i in range(9):
            row = []
            for j in range(9):
                value = self.grid_entries[i][j].get().strip()
                if value and value.isdigit():
                    row.append(int(value))
                else:
                    row.append(0)
            grid.append(row)
        
        self.original_grid = grid
        # 移除重复的网格显示，避免与识别结果重复
        self.update_status("已获取当前网格状态")
    
    def select_area(self):
        """开始区域选择"""
        try:
            self.update_status("正在启动区域选择...")
            self.root.withdraw()  # 隐藏主窗口
            self.area_selector.start_selection()
            self.root.after(100, self.check_selection_done)
        except Exception as e:
            self.root.deiconify()  # 确保主窗口显示
            messagebox.showerror("错误", f"启动区域选择失败: {str(e)}")
            self.update_status("区域选择启动失败")
    
    def check_selection_done(self):
        """检查选择是否完成"""
        try:
            if self.area_selector.selection_window is not None:
                # 选择窗口仍然存在，继续检查
                self.root.after(100, self.check_selection_done)
            else:
                # 选择完成或取消，显示主窗口
                self.root.deiconify()
                if not hasattr(self, '_area_selected_recently'):
                    self.update_status("区域选择已取消")
        except Exception as e:
            # 出现异常，确保主窗口显示
            self.root.deiconify()
            self.update_status("区域选择出现错误")
        
    def setup_ui(self):
        """设置用户界面 - 竖置滚轮滑动细长界面"""
        # 创建主滚动框架
        self.canvas = tk.Canvas(self.root)
        self.scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # 绑定鼠标滚轮事件
        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        self.canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # 布局滚动组件
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # 主框架
        main_frame = ttk.Frame(self.scrollable_frame, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 行1：输出信息（左右布局："输出：" + 内容）
        output_frame = ttk.Frame(main_frame)
        output_frame.pack(fill=tk.X, pady=(0, 5))
        
        # 左侧："输出："标签
        tk.Label(output_frame, text="输出：", font=('Arial', 9)).pack(side=tk.LEFT, padx=(0, 5))
        
        # 右侧：输出内容
        self.status_var = tk.StringVar(value="就绪")
        self.info_text = tk.Label(output_frame, textvariable=self.status_var, fg='black', 
                                bg='white', relief=tk.SUNKEN, bd=1, anchor='w', justify='left')
        self.info_text.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 0), ipady=3)
        
        # 行2：进度显示（左右布局：进度条 + 百分比）
        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill=tk.X, pady=(0, 5))
        
        # 左侧：进度条
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, 
                                          maximum=100, mode='determinate')
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        # 右侧：进度百分比
        self.progress_label = tk.Label(progress_frame, text="0%", width=5, anchor='e')
        self.progress_label.pack(side=tk.RIGHT)
        
        # 行2：识别游戏地图 选择游戏区域 设置
        row2_frame = ttk.Frame(main_frame)
        row2_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(row2_frame, text="2识别地图", command=self.capture_and_recognize, width=12).pack(side=tk.LEFT, padx=(0, 5), anchor='w')
        ttk.Button(row2_frame, text="1选择区域", command=self.select_area, width=12).pack(side=tk.LEFT, padx=(0, 5), anchor='w')
        
        # 创建按钮样式
        style = ttk.Style()
        style.configure("Green.TButton", background="#4CAF50", foreground="blue")
        style.configure("Red.TButton", background="#f44336", foreground="red")
        
        self.settings_button = ttk.Button(row2_frame, text="展开参数设置", command=self.toggle_settings, style="Green.TButton")
        self.settings_button.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # 行3：求解 上一解 下一解
        row3_frame = ttk.Frame(main_frame)
        row3_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(row3_frame, text="3求数独解", command=self.solve_sudoku, width=12).pack(side=tk.LEFT, padx=(0, 5), anchor='w')
        ttk.Button(row3_frame, text="上一解", command=self.prev_solution, width=12).pack(side=tk.LEFT, padx=(0, 5), anchor='w')
        ttk.Button(row3_frame, text="下一解", command=self.next_solution, width=12).pack(side=tk.LEFT, padx=(0, 5), anchor='w')
        
        # 行4：输出解 输出延迟（ms）输入框 终止按钮
        row4_frame = ttk.Frame(main_frame)
        row4_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(row4_frame, text="4键鼠输出", command=self.auto_fill_external, width=12).pack(side=tk.LEFT, padx=(0, 5), anchor='w')
        ttk.Label(row4_frame, text="输出延迟（ms）:").pack(side=tk.LEFT, padx=2)
        self.delay_var = tk.StringVar(value="100")
        delay_entry = ttk.Entry(row4_frame, textvariable=self.delay_var, width=8)
        delay_entry.pack(side=tk.LEFT, padx=(0, 5))
        
        # 红色圆形终止按钮 - 使用Canvas创建真正的圆形
        self.stop_button_frame = tk.Frame(row4_frame, width=30, height=30)
        self.stop_button_frame.pack(side=tk.LEFT, padx=(5, 0))
        self.stop_button_frame.pack_propagate(False)
        
        self.stop_canvas = tk.Canvas(self.stop_button_frame, width=30, height=30, 
                                   highlightthickness=0, bg='SystemButtonFace')
        self.stop_canvas.pack()
        
        # 绘制红色圆形
        self.stop_circle = self.stop_canvas.create_oval(2, 2, 28, 28, fill='red', outline='darkred', width=2)
        # 添加白色停止符号
        self.stop_symbol = self.stop_canvas.create_text(15, 15, text="■", fill='white', font=('Arial', 10, 'bold'))
        
        # 绑定点击事件
        self.stop_canvas.bind("<Button-1>", lambda e: self.stop_current_operation())
        # 添加鼠标悬停效果
        self.stop_canvas.bind("<Enter>", lambda e: self.stop_canvas.itemconfig(self.stop_circle, fill='darkred'))
        self.stop_canvas.bind("<Leave>", lambda e: self.stop_canvas.itemconfig(self.stop_circle, fill='red'))
        
        # 行5：从识别填充 从求解填充 清空编辑器
        row5_frame = ttk.Frame(main_frame)
        row5_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(row5_frame, text="从识别填充", command=self.fill_from_recognition).pack(side=tk.LEFT, padx=(0, 5), fill=tk.X, expand=True)
        ttk.Button(row5_frame, text="从求解填充", command=self.auto_fill).pack(side=tk.LEFT, padx=(0, 5), fill=tk.X, expand=True)
        ttk.Button(row5_frame, text="清空编辑器", command=self.clear_grid).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # 行6-行15：数独编辑器区域（9X9方格）
        grid_frame = ttk.LabelFrame(main_frame, text="数独编辑器", padding="10")
        grid_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.create_sudoku_grid(grid_frame)
        
        # 解的选择显示
        solution_frame = ttk.Frame(main_frame)
        solution_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.solution_var = tk.StringVar(value="解 1 / 0")
        ttk.Label(solution_frame, textvariable=self.solution_var, font=('Arial', 10, 'bold')).pack()
        
        # JSON文件选择区域（在"解 1/0"下方）
        json_frame = ttk.Frame(main_frame)
        json_frame.pack(fill=tk.X, pady=(5, 10))
        
        ttk.Label(json_frame, text="配置文件").pack(side=tk.LEFT)
        
        # 获取当前目录下的所有json文件
        self.json_files = self.get_json_files()
        self.selected_json = tk.StringVar(value=self.json_files[0] if self.json_files else "无可用文件")
        
        json_combobox = ttk.Combobox(json_frame, textvariable=self.selected_json, 
                                   values=self.json_files, state="readonly", width=20)
        json_combobox.pack(side=tk.LEFT, padx=(5, 5))
        
        ttk.Button(json_frame, text="加载", command=self.load_selected_config).pack(side=tk.LEFT)
        
        # 初始化设置变量
        self.thread_count = tk.IntVar(value=getattr(self, 'thread_count_value', 4))
        self.outer_line_width = tk.IntVar(value=getattr(self, 'outer_line_width_value', 30))
        self.inner_line_width = tk.IntVar(value=getattr(self, 'inner_line_width_value', 10))
        self.thick_line_width = tk.IntVar(value=getattr(self, 'thick_line_width_value', 20))
        
        # 创建可折叠的设置区域
        self.create_settings_area(main_frame)
    
    def create_settings_area(self, parent):
        """创建可折叠的设置区域"""
        # 设置区域框架（初始隐藏）
        self.settings_frame = ttk.LabelFrame(parent, text="", padding="0")
        # 不立即pack，等待展开时再显示
        
        # 行1：识别线程数滑动条
        thread_frame = ttk.Frame(self.settings_frame)
        thread_frame.pack(fill=tk.X, pady=(0, 0))
        
        ttk.Label(thread_frame, text="识别线程数:").pack(side=tk.LEFT)
        thread_scale = tk.Scale(thread_frame, from_=2, to=32, variable=self.thread_count, 
                               orient=tk.HORIZONTAL, resolution=2, length=200)
        thread_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        
        self.thread_label = ttk.Label(thread_frame, text=str(self.thread_count.get()), width=3)
        self.thread_label.pack(side=tk.RIGHT)
        
        thread_scale.configure(command=lambda v: self.thread_label.config(text=str(int(float(v)))))
        
        # 行2：识别框外框宽度滑动条
        outer_frame = ttk.Frame(self.settings_frame)
        outer_frame.pack(fill=tk.X, pady=(0, 1))
        
        ttk.Label(outer_frame, text="外框线宽度:").pack(side=tk.LEFT)
        outer_scale = tk.Scale(outer_frame, from_=1, to=50, variable=self.outer_line_width, 
                              orient=tk.HORIZONTAL, resolution=1, length=200)
        outer_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        
        self.outer_label = ttk.Label(outer_frame, text=str(self.outer_line_width.get()), width=3)
        self.outer_label.pack(side=tk.RIGHT)
        
        outer_scale.configure(command=lambda v: self.outer_label.config(text=str(int(float(v)))))
        
        # 行3：识别框内框宽度滑动条
        inner_frame = ttk.Frame(self.settings_frame)
        inner_frame.pack(fill=tk.X, pady=(0, 1))
        
        ttk.Label(inner_frame, text="内框线宽度:").pack(side=tk.LEFT)
        inner_scale = tk.Scale(inner_frame, from_=1, to=30, variable=self.inner_line_width, 
                              orient=tk.HORIZONTAL, resolution=1, length=200)
        inner_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        
        self.inner_label = ttk.Label(inner_frame, text=str(self.inner_line_width.get()), width=3)
        self.inner_label.pack(side=tk.RIGHT)
        
        inner_scale.configure(command=lambda v: self.inner_label.config(text=str(int(float(v)))))
        
        # 行4：识别框粗线宽度滑动条
        thick_frame = ttk.Frame(self.settings_frame)
        thick_frame.pack(fill=tk.X, pady=(0, 1))
        
        ttk.Label(thick_frame, text="九宫分界线宽:").pack(side=tk.LEFT)
        thick_scale = tk.Scale(thick_frame, from_=1, to=40, variable=self.thick_line_width, 
                              orient=tk.HORIZONTAL, resolution=1, length=200)
        thick_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        
        self.thick_label = ttk.Label(thick_frame, text=str(self.thick_line_width.get()), width=3)
        self.thick_label.pack(side=tk.RIGHT)
        
        thick_scale.configure(command=lambda v: self.thick_label.config(text=str(int(float(v)))))
        
        # 行5：游戏识别区域设置
        area_frame = ttk.LabelFrame(self.settings_frame, text="识别区域", padding="3")
        area_frame.pack(fill=tk.X, pady=(0, 2))
        
        # 起点坐标
        start_frame = ttk.Frame(area_frame)
        start_frame.pack(fill=tk.X, pady=(0, 1))
        
        ttk.Label(start_frame, text="起点X:").pack(side=tk.LEFT)
        ttk.Entry(start_frame, textvariable=self.x_var, width=8).pack(side=tk.LEFT, padx=(5, 10))
        ttk.Label(start_frame, text="起点Y:").pack(side=tk.LEFT)
        ttk.Entry(start_frame, textvariable=self.y_var, width=8).pack(side=tk.LEFT, padx=(5, 0))
        
        # 宽度高度
        size_frame = ttk.Frame(area_frame)
        size_frame.pack(fill=tk.X, pady=(1, 0))
        
        ttk.Label(size_frame, text="宽度:").pack(side=tk.LEFT)
        ttk.Entry(size_frame, textvariable=self.width_var, width=8).pack(side=tk.LEFT, padx=(5, 10))
        ttk.Label(size_frame, text="高度:").pack(side=tk.LEFT)
        ttk.Entry(size_frame, textvariable=self.height_var, width=8).pack(side=tk.LEFT, padx=(5, 0))
        
        # 最下方：三个按钮同行布局
        bottom_frame = ttk.Frame(self.settings_frame)
        bottom_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(bottom_frame, text="预览截图图像", command=self.show_screenshot).pack(side=tk.LEFT, padx=(0, 5), fill=tk.X, expand=True)
        ttk.Button(bottom_frame, text="预览识别区域", command=self.show_recognition_regions).pack(side=tk.LEFT, padx=(0, 5), fill=tk.X, expand=True)
        ttk.Button(bottom_frame, text="导出json文件", command=self.save_config_to_file).pack(side=tk.LEFT, fill=tk.X, expand=True)
    
    def toggle_settings(self):
        """切换设置区域的显示/隐藏"""
        # 获取样式对象
        style = ttk.Style()
        
        if self.settings_expanded:
            # 收起设置
            self.settings_frame.pack_forget()
            self.root.geometry("500x750")
            # 设置为绿框蓝字
            style.configure("Green.TButton", background="#4CAF50", foreground="blue")
            self.settings_button.configure(text="展开参数设置", style="Green.TButton")
            self.settings_expanded = False
        else:
            # 展开设置
            self.settings_frame.pack(fill=tk.X, pady=(10, 0))
            self.root.geometry("500x1150")
            # 设置为红框红字
            style.configure("Red.TButton", background="#f44336", foreground="red")
            self.settings_button.configure(text="关闭参数设置", style="Red.TButton")
            self.settings_expanded = True
    
    def get_json_files(self):
        """获取当前目录下的所有json文件"""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            json_files = [f for f in os.listdir(current_dir) if f.endswith('.json')]
            return json_files if json_files else ["无可用文件"]
        except Exception as e:
            print(f"获取json文件列表失败: {str(e)}")
            return ["无可用文件"]
    
    def load_selected_config(self):
        """加载选中的配置文件"""
        selected_file = self.selected_json.get()
        if selected_file == "无可用文件":
            messagebox.showwarning("警告", "没有可用的配置文件")
            return
        
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(current_dir, selected_file)
            
            with open(file_path, 'r') as f:
                config = json.load(f)
                self.x_var.set(str(config.get('x', 100)))
                self.y_var.set(str(config.get('y', 100)))
                self.width_var.set(str(config.get('width', 450)))
                self.height_var.set(str(config.get('height', 450)))
                self.thread_count.set(config.get('thread_count', 4))
                self.outer_line_width.set(config.get('outer_line_width', 25))
                self.inner_line_width.set(config.get('inner_line_width', 15))
                self.thick_line_width.set(config.get('thick_line_width', 20))
                
                # 更新标签显示
                if hasattr(self, 'thread_label'):
                    self.thread_label.config(text=str(self.thread_count.get()))
                if hasattr(self, 'outer_label'):
                    self.outer_label.config(text=str(self.outer_line_width.get()))
                if hasattr(self, 'inner_label'):
                    self.inner_label.config(text=str(self.inner_line_width.get()))
                if hasattr(self, 'thick_label'):
                    self.thick_label.config(text=str(self.thick_line_width.get()))
                
                self.update_status(f"配置已从 {selected_file} 加载")
                print(f"配置已从文件加载: {file_path}")
        except Exception as e:
            messagebox.showerror("错误", f"加载配置文件失败: {str(e)}")
            self.update_status("配置加载失败")
    
    def load_config_from_file(self):
        """从文件加载配置"""
        from tkinter import filedialog
        file_path = filedialog.askopenfilename(
            title="选择配置文件",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=os.path.dirname(os.path.abspath(self.config_file))
        )
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    config = json.load(f)
                    self.x_var.set(str(config.get('x', 100)))
                    self.y_var.set(str(config.get('y', 100)))
                    self.width_var.set(str(config.get('width', 450)))
                    self.height_var.set(str(config.get('height', 450)))
                    self.thread_count.set(config.get('thread_count', 4))
                    self.outer_line_width.set(config.get('outer_line_width', 25))
                    self.inner_line_width.set(config.get('inner_line_width', 15))
                    self.thick_line_width.set(config.get('thick_line_width', 20))
                    
                    # 更新标签显示
                    self.thread_label.config(text=str(self.thread_count.get()))
                    self.outer_label.config(text=str(self.outer_line_width.get()))
                    self.inner_label.config(text=str(self.inner_line_width.get()))
                    self.thick_label.config(text=str(self.thick_line_width.get()))
                    
                    self.update_status(f"配置已从 {os.path.basename(file_path)} 加载")
                    print(f"配置已从文件加载: {file_path}")
            except Exception as e:
                messagebox.showerror("错误", f"加载配置文件失败: {str(e)}")
                self.status_var.set("配置加载失败")
    
    def save_config_to_file(self):
        """保存配置到文件"""
        from tkinter import filedialog
        file_path = filedialog.asksaveasfilename(
            title="保存配置文件",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=os.path.dirname(os.path.abspath(self.config_file)),
            initialfile="sudoku_config.json"
        )
        if file_path:
            try:
                config = {
                    'x': int(self.x_var.get()),
                    'y': int(self.y_var.get()),
                    'width': int(self.width_var.get()),
                    'height': int(self.height_var.get()),
                    'thread_count': self.thread_count.get(),
                    'outer_line_width': self.outer_line_width.get(),
                    'inner_line_width': self.inner_line_width.get(),
                    'thick_line_width': self.thick_line_width.get()
                }
                with open(file_path, 'w') as f:
                    json.dump(config, f, indent=2)
                self.status_var.set(f"配置已保存到 {os.path.basename(file_path)}")
                print(f"配置已保存到文件: {file_path}")
            except Exception as e:
                messagebox.showerror("错误", f"保存配置文件失败: {str(e)}")
                self.status_var.set("配置保存失败")
    

        

    
    def show_screenshot(self):
        """显示截图识别图像"""
        try:
            x = int(self.x_var.get())
            y = int(self.y_var.get())
            width = int(self.width_var.get())
            height = int(self.height_var.get())
            
            print(f"正在截取区域: ({x}, {y}) 大小: {width}x{height}")
            
            # 截取图像
            image = self.recognizer.capture_screen_area(x, y, width, height)
            
            # 保存临时图像
            import tempfile
            import os
            temp_file = os.path.join(tempfile.gettempdir(), 'sudoku_screenshot.png')
            cv2.imwrite(temp_file, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
            # 显示图像窗口
            self.show_image_window(temp_file, "截图识别图像")
            
        except Exception as e:
            print(f"显示截图失败: {str(e)}")
            messagebox.showerror("错误", f"显示截图失败: {str(e)}")
    
    def show_image_window(self, image_path, title):
        """显示图像窗口"""
        try:
            image_window = tk.Toplevel(self.root)
            image_window.title(title)
            image_window.geometry("500x700")
            image_window.resizable(True, True)
            # 普通窗口，可以正常关闭
            image_window.protocol("WM_DELETE_WINDOW", image_window.destroy)
            image_window.bind('<Escape>', lambda e: image_window.destroy())
            
            from PIL import Image, ImageTk
            pil_image = Image.open(image_path)
            
            # 调整图像大小
            max_size = 550
            pil_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(pil_image)
            
            image_label = ttk.Label(image_window, image=photo)
            image_label.image = photo
            image_label.pack(pady=20)
            
            ttk.Button(image_window, text="关闭", command=image_window.destroy).pack(pady=10)
            
        except Exception as e:
            print(f"显示图像窗口失败: {str(e)}")
            messagebox.showerror("错误", f"显示图像窗口失败: {str(e)}")
    

    
    def on_thread_count_change(self, value=None):
        """线程数改变时的回调"""
        # 更新显示标签
        self.thread_label.config(text=str(int(self.thread_count.get())))
        self.save_config()
    
    def on_line_width_change(self, event=None):
        """线框宽度变化回调"""
        # 更新区域选择器的线宽设置
        self.area_selector.outer_line_width = self.outer_line_width.get()
        self.area_selector.inner_line_width = self.inner_line_width.get()
        self.area_selector.thick_line_width = self.thick_line_width.get()
        self.save_config()
    
    def update_progress(self, progress):
        """更新进度条和百分比"""
        self.progress_var.set(progress)
        if hasattr(self, 'progress_label'):
            self.progress_label.config(text=f"{int(progress)}%")
        self.root.update_idletasks()
    
    def reset_settings(self):
        """重置所有设置到默认值"""
        self.thread_count.set(4)
        self.thread_label.config(text="4")
        self.outer_line_width.set(25)
        self.inner_line_width.set(15)
        self.thick_line_width.set(20)
        # 更新区域选择器的线宽设置
        if hasattr(self, 'area_selector'):
            self.area_selector.outer_line_width = 25
            self.area_selector.inner_line_width = 15
            self.area_selector.thick_line_width = 20
        self.progress_var.set(0)
        self.status_var.set("设置已重置")
        self.save_config()
        # 2秒后恢复就绪状态
        self.root.after(2000, lambda: self.status_var.set("就绪"))
    
    def show_recognition_regions(self):
        """显示单元格识别区域的可视化"""
        try:
            x = int(self.x_var.get())
            y = int(self.y_var.get())
            width = int(self.width_var.get())
            height = int(self.height_var.get())
            
            if width < 100 or height < 100:
                messagebox.showwarning("警告", "截图区域太小，请设置合适的宽度和高度")
                return
            
            self.status_var.set("准备截图，请切换到数独界面...")
            self.root.update()
            
            def visualize_thread():
                try:
                    time.sleep(1)  # 给用户时间切换界面
                    
                    self.root.after(0, lambda: self.status_var.set("正在截图..."))
                    image = self.recognizer.capture_screen_area(x, y, width, height)
                    
                    self.root.after(0, lambda: self.status_var.set("正在生成识别区域可视化..."))
                    
                    # 预处理图像
                    processed_image = self.recognizer.preprocess_image(image)
                    
                    # 获取当前线框宽度设置
                    line_widths = {
                        'outer': self.outer_line_width.get(),
                        'inner': self.inner_line_width.get(),
                        'thick': self.thick_line_width.get()
                    }
                    
                    # 生成可视化图像
                    vis_image = self.recognizer.visualize_cell_regions(processed_image, line_widths)
                    
                    # 保存可视化图像
                    import tempfile
                    import os
                    temp_file = os.path.join(tempfile.gettempdir(), 'sudoku_regions.png')
                    cv2.imwrite(temp_file, vis_image)
                    
                    # 在主线程中显示结果
                    self.root.after(0, lambda: self.show_visualization_window(temp_file, line_widths))
                    
                except Exception as e:
                    self.root.after(0, lambda: self.on_visualization_error(str(e)))
            
            threading.Thread(target=visualize_thread, daemon=True).start()
            
        except ValueError:
            messagebox.showerror("错误", "请输入有效的数字坐标")
            self.status_var.set("参数错误")
        except Exception as e:
            messagebox.showerror("错误", f"启动可视化失败: {str(e)}")
            self.status_var.set("启动失败")
    
    def show_visualization_window(self, image_path, line_widths):
        """显示识别区域可视化窗口"""
        try:
            # 创建新窗口
            vis_window = tk.Toplevel(self.root)
            vis_window.title("单元格识别区域可视化")
            vis_window.geometry("800x700")
            vis_window.resizable(True, True)
            # 普通窗口，可以正常关闭
            vis_window.protocol("WM_DELETE_WINDOW", vis_window.destroy)
            vis_window.bind('<Escape>', lambda e: vis_window.destroy())
            
            # 添加说明文本
            info_frame = ttk.Frame(vis_window, padding="10")
            info_frame.pack(fill=tk.X)
            
            info_text = f"""识别区域可视化说明：
                            • 绿色框：每个单元格的实际识别区域
                            • 数字标注：单元格坐标 (行,列)
                            • 当前线框设置：外框={line_widths['outer']}px, 内部={line_widths['inner']}px, 分割={line_widths['thick']}px
                            • 识别区域应完全避开网格线，只包含数字内容"""
            
            ttk.Label(info_frame, text=info_text, font=("Arial", 9)).pack(anchor=tk.W)
            
            # 显示图像
            from PIL import Image, ImageTk
            pil_image = Image.open(image_path)
            
            # 调整图像大小以适应窗口
            max_width, max_height = 750, 500
            pil_image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(pil_image)
            
            image_label = ttk.Label(vis_window, image=photo)
            image_label.image = photo  # 保持引用
            image_label.pack(pady=10)
            
            # 添加操作按钮
            button_frame = ttk.Frame(vis_window)
            button_frame.pack(fill=tk.X, padx=10, pady=10)
            
            ttk.Button(button_frame, text="保存图像", 
                      command=lambda: self.save_visualization(image_path)).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="关闭", 
                      command=vis_window.destroy).pack(side=tk.RIGHT, padx=5)
            
            self.status_var.set("识别区域可视化完成")
            
        except Exception as e:
            messagebox.showerror("错误", f"显示可视化失败: {str(e)}")
            self.status_var.set("显示失败")
    
    def save_visualization(self, temp_path):
        """保存可视化图像"""
        try:
            from tkinter import filedialog
            save_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
                title="保存识别区域可视化图像"
            )
            if save_path:
                import shutil
                shutil.copy2(temp_path, save_path)
                messagebox.showinfo("成功", f"图像已保存到: {save_path}")
        except Exception as e:
            messagebox.showerror("错误", f"保存失败: {str(e)}")
    
    def on_visualization_error(self, error_msg):
        """可视化错误回调"""
        messagebox.showerror("错误", f"生成可视化失败: {error_msg}")
        self.status_var.set("可视化失败")
        
    def capture_and_recognize(self):
        """截图并识别数独"""
        try:
            x = int(self.x_var.get())
            y = int(self.y_var.get())
            width = int(self.width_var.get())
            height = int(self.height_var.get())
            
            if width < 100 or height < 100:
                messagebox.showwarning("警告", "截图区域太小，请设置合适的宽度和高度")
                return
            
            # 重置进度条
            self.progress_var.set(0)
            self.status_var.set("准备截图，请切换到数独界面...")
            self.root.update()
            
            # 使用线程避免界面卡死
            def capture_thread():
                try:
                    time.sleep(1)  # 给用户时间切换界面
                    
                    # 开始计时
                    self.recognition_start_time = time.time()
                    recognition_start_time = self.recognition_start_time
                    
                    self.root.after(0, lambda: self.status_var.set("正在截图..."))
                    image = self.recognizer.capture_screen_area(x, y, width, height)
                    
                    self.root.after(0, lambda: self.status_var.set("正在识别数字..."))
                    
                    # 修改recognizer以支持多线程
                    import concurrent.futures
                    thread_count = self.thread_count.get()
                    
                    # 创建进度回调函数
                    def progress_callback(progress):
                        # 检查是否被用户终止
                        if hasattr(self, 'operation_stopped') and self.operation_stopped:
                            return False  # 返回False表示终止操作
                        self.root.after(0, lambda: self.update_progress(progress))
                        return True  # 返回True表示继续操作
                    
                    # 详细信息回调函数
                    def detailed_callback(row, col, digit, confidence_dict):
                        # 检查是否被用户终止
                        if hasattr(self, 'operation_stopped') and self.operation_stopped:
                            return False  # 返回False表示终止操作
                        
                        # 更新GUI状态显示
                        max_confidence = max(confidence_dict.values()) if confidence_dict else 0
                        status_text = f"{row+1}行{col+1}列：{digit if digit != 0 else '空'}（{max_confidence:.0f}%）"
                        self.root.after(0, lambda: self.status_var.set(status_text))
                        
                        # 增强终端输出：显示更多识别详情
                        if digit != 0:  # 只输出非空格单元格
                            max_conf = max(confidence_dict.values()) if confidence_dict else 0
                            sorted_confs = sorted(confidence_dict.items(), key=lambda x: x[1], reverse=True)
                            
                            # 输出条件：
                            # 1. 置信度低于80%
                            # 2. 有多个数字置信度接近（差距小于30%）
                            # 3. 识别结果可能有误差的情况
                            should_output = (max_conf < 80 or 
                                           (len(sorted_confs) >= 2 and sorted_confs[0][1] - sorted_confs[1][1] < 30))
                            
                            if should_output:
                                # 显示前3个最高置信度的数字
                                top_candidates = sorted_confs[:3]
                                candidates_str = " ".join([f"{num}({conf:.0f}%)" for num, conf in top_candidates if conf > 10])
                                print(f"第{row+1}行{col+1}列识别为{digit}，候选：{candidates_str}")
                            elif max_conf < 95:  # 中等置信度也显示简要信息
                                print(f"第{row+1}行{col+1}列：{digit}({max_conf:.0f}%)")
                    
                    # 获取当前线框宽度设置
                    line_widths = {
                        'outer': self.outer_line_width.get(),
                        'inner': self.inner_line_width.get(),
                        'thick': self.thick_line_width.get()
                    }
                    
                    print("\n开始识别数独，详细置信度信息：")
                    # 使用多线程识别，传递线框宽度参数和详细回调（传递终止标志）
                    self.original_grid = self.recognizer.recognize_sudoku_with_threads(
                        image, thread_count, progress_callback, line_widths, detailed_callback, self
                    )
                    
                    # 检查是否被用户终止
                    if self.original_grid is None:
                        self.root.after(0, lambda: self.status_var.set("识别已被用户终止"))
                        self.root.after(0, lambda: self.progress_var.set(0))
                        return
                    
                    # 计算单元格位置
                    cell_width = width // 9
                    cell_height = height // 9
                    self.cell_positions = self.auto_filler.calculate_cell_positions(x, y, cell_width, cell_height)
                    
                    # 记录识别时间统计
                    recognition_time = time.time() - recognition_start_time
                    perf_config.stats['recognition_time'].append(recognition_time)
                    print(f"识别耗时: {recognition_time:.2f}秒")
                    
                    # 在主线程中更新UI
                    self.root.after(0, lambda: self.on_recognition_complete(self.original_grid))
                    
                except Exception as e:
                    self.root.after(0, lambda: self.on_recognition_error(str(e)))
            
            threading.Thread(target=capture_thread, daemon=True).start()
            
        except ValueError:
            messagebox.showerror("错误", "请输入有效的数字坐标")
            self.status_var.set("参数错误")
        except Exception as e:
            messagebox.showerror("错误", f"启动识别失败: {str(e)}")
            self.status_var.set("启动失败")
    
    def on_recognition_complete(self, grid):
        """识别完成回调"""
        self.progress_var.set(100)  # 确保进度条显示100%
        
        # 获取识别时长并显示到第一行
        if hasattr(self, 'recognition_start_time'):
            recognition_time = time.time() - self.recognition_start_time
            time_msg = f"识别完成，耗时: {recognition_time:.2f}秒"
            self.status_var.set(time_msg)
            print(time_msg)
        else:
            status_msg = "识别完成"
            self.status_var.set(status_msg)
            print(status_msg)
        
        self.display_grid(grid, "识别的数独:")
        self.fill_from_recognition()  # 自动填充到编辑器
        
        # 立即重置进度条
        self.root.after(1000, lambda: self.progress_var.set(0))
    
    def on_recognition_error(self, error_msg):
        """识别错误回调"""
        messagebox.showerror("错误", f"截图识别失败: {error_msg}")
        self.status_var.set("识别失败")
    
    def solve_sudoku(self):
        """求解数独"""
        # 首先获取当前编辑器中的数独
        self.get_current_grid()
        
        if self.original_grid is None:
            messagebox.showwarning("警告", "请先输入数独或截图识别")
            return
        
        # 检查数独是否有效
        if not self.is_valid_sudoku(self.original_grid):
            messagebox.showerror("错误", "当前数独存在冲突，请检查输入")
            return
        
        self.status_var.set("正在求解...")
        self.root.update()
        
        def solve_thread():
            try:
                print("\n开始求解数独...")
                # 开始计时
                solve_start_time = time.time()
                
                # 复制原始网格
                grid_copy = [row[:] for row in self.original_grid]
                
                # 显示原始数独
                print("原始数独：")
                for i, row in enumerate(grid_copy):
                    row_str = " ".join([str(cell) if cell != 0 else '.' for cell in row])
                    print(f"第{i+1}行：{row_str}")
                
                # 更新状态显示求解进度
                self.root.after(0, lambda: self.status_var.set("正在分析数独结构..."))
                self.root.after(0, lambda: self.progress_var.set(10))
                
                # 创建进度更新回调
                def progress_update(message):
                    # 检查终止标志
                    if hasattr(self, 'operation_stopped') and self.operation_stopped:
                        return False
                    self.root.after(0, lambda: self.status_var.set(message))
                    return True
                
                # 性能优化：使用优化的求解算法，限制最大解数量（传递终止标志）
                solutions = self.solver.solve_all(grid_copy, progress_update, max_solutions=50, stop_flag=self)
                
                # 检查是否被终止
                if hasattr(self, 'operation_stopped') and self.operation_stopped:
                    self.root.after(0, lambda: self.status_var.set("求解已终止"))
                    self.root.after(0, lambda: self.progress_var.set(0))
                    return
                
                # 记录求解时间统计
                solve_time = time.time() - solve_start_time
                perf_config.stats['solve_time'].append(solve_time)
                
                # 显示求解结果信息
                if solutions:
                    print(f"\n求解完成！找到 {len(solutions)} 个解 (耗时: {solve_time:.2f}秒)")
                    for idx, solution in enumerate(solutions[:3]):  # 最多显示前3个解
                        print(f"\n解 {idx+1}：")
                        for i, row in enumerate(solution):
                            row_str = " ".join([str(cell) for cell in row])
                            print(f"第{i+1}行：{row_str}")
                        if idx < len(solutions) - 1 and idx < 2:
                            print("-" * 30)
                    if len(solutions) > 3:
                        print(f"\n... 还有 {len(solutions) - 3} 个解")
                else:
                    print(f"\n求解完成！无解 (耗时: {solve_time:.2f}秒)")
                
                # 完成进度条
                self.root.after(0, lambda: self.progress_var.set(100))
                
                # 在主线程中更新UI
                self.root.after(0, lambda: self.on_solve_complete(solutions))
                
            except Exception as e:
                print(f"\n求解失败：{str(e)}")
                self.root.after(0, lambda: self.progress_var.set(0))
                self.root.after(0, lambda: self.on_solve_error(str(e)))
        
        threading.Thread(target=solve_thread, daemon=True).start()
    
    def is_valid_sudoku(self, grid):
        """检查数独是否有效（无冲突）"""
        for i in range(9):
            for j in range(9):
                if grid[i][j] != 0:
                    num = grid[i][j]
                    # 临时移除当前数字
                    grid[i][j] = 0
                    # 检查是否有效
                    if not self.solver.is_valid(grid, i, j, num):
                        grid[i][j] = num  # 恢复
                        return False
                    grid[i][j] = num  # 恢复
        return True
    
    def on_solve_complete(self, solutions):
        """求解完成回调"""
        self.solutions = solutions
        if self.solutions:
            self.current_solution = 0
            self.solution_var.set(f"解 {self.current_solution + 1} / {len(self.solutions)}")
            self.display_grid(self.solutions[self.current_solution], f"解 {self.current_solution + 1}:")
            self.status_var.set(f"找到 {len(self.solutions)} 个解")
            # 自动填充到编辑器
            self.auto_fill_to_editor()
        else:
            messagebox.showinfo("结果", "无解")
            self.status_var.set("无解")
        # 重置进度条
        self.progress_var.set(0)
    
    def on_solve_error(self, error_msg):
        """求解错误回调"""
        messagebox.showerror("错误", f"求解失败: {error_msg}")
        self.status_var.set("求解失败")
    
    def prev_solution(self):
        """显示上一个解"""
        if self.solutions and self.current_solution > 0:
            self.current_solution -= 1
            self.solution_var.set(f"解 {self.current_solution + 1} / {len(self.solutions)}")
            self.display_grid(self.solutions[self.current_solution], f"解 {self.current_solution + 1}:")
            # 自动填充到编辑器
            self.auto_fill_to_editor()
    
    def next_solution(self):
        """显示下一个解"""
        if self.solutions and self.current_solution < len(self.solutions) - 1:
            self.current_solution += 1
            self.solution_var.set(f"解 {self.current_solution + 1} / {len(self.solutions)}")
            self.display_grid(self.solutions[self.current_solution], f"解 {self.current_solution + 1}:")
            # 自动填充到编辑器
            self.auto_fill_to_editor()
    
    def auto_fill_to_editor(self):
        """自动填充数独到编辑器（内部方法）"""
        if not self.solutions or self.current_solution >= len(self.solutions):
            return
        
        try:
            solution = self.solutions[self.current_solution]
            
            # 将解填充到编辑器网格中
            for i in range(9):
                for j in range(9):
                    entry = self.grid_entries[i][j]
                    entry.delete(0, tk.END)
                    if solution[i][j] != 0:
                        entry.insert(0, str(solution[i][j]))
                        # 如果是原始识别的数字，保持黑色；如果是求解的数字，设为绿色
                        if self.original_grid and self.original_grid[i][j] != 0:
                            entry.config(fg='black')  # 识别的数字保持黑色
                        else:
                            entry.config(fg='green')  # 求解的数字设为绿色
                    else:
                        entry.config(fg='black')  # 空格设为黑色
            
            # 检查冲突并标红
            self.check_conflicts()
            
        except Exception as e:
            print(f"自动填充失败: {str(e)}")
    
    def auto_fill(self):
        """手动触发的自动填充数独到编辑器"""
        if not self.solutions or self.current_solution >= len(self.solutions):
            messagebox.showwarning("警告", "请先求解数独")
            return
        
        self.status_var.set("正在自动填充...")
        self.root.update()
        
        self.auto_fill_to_editor()
        
        status_msg = "自动填充完成"
        self.status_var.set(status_msg)
        print(status_msg)
        messagebox.showinfo("完成", "解已填充到编辑器中")
    
    def auto_fill_external(self):
        """自动填充到外部程序（原功能）"""
        if not self.solutions or self.current_solution >= len(self.solutions):
            messagebox.showwarning("警告", "请先求解数独")
            return
        
        if self.cell_positions is None:
            # 如果没有位置信息，尝试使用当前配置计算位置
            try:
                x = int(self.x_var.get())
                y = int(self.y_var.get())
                width = int(self.width_var.get())
                height = int(self.height_var.get())
                
                if width < 100 or height < 100:
                    messagebox.showwarning("警告", "请先设置正确的截图区域坐标，或重新识别数独")
                    return
                
                # 计算单元格位置
                cell_width = width // 9
                cell_height = height // 9
                self.cell_positions = self.auto_filler.calculate_cell_positions(x, y, cell_width, cell_height)
                
            except ValueError:
                messagebox.showwarning("警告", "请先识别数独以获取位置信息，或设置正确的截图区域坐标")
                return
        
        # 直接开始填充，不显示确认对话框以避免GUI消失
        self.status_var.set("准备自动填充...")
        self.root.update()
        
        def fill_thread():
            try:
                # 获取延迟时间
                try:
                    delay = int(self.delay_var.get()) / 1000.0
                except:
                    delay = 0.1
                
                time.sleep(3)  # 给用户时间切换界面
                current_solution_grid = self.solutions[self.current_solution]
                self.auto_filler.fill_solution(self.original_grid, current_solution_grid, self.cell_positions, delay)
                def update_status():
                    status_msg = "外部填充完成"
                    self.status_var.set(status_msg)
                    print(status_msg)
                self.root.after(0, update_status)
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda: self.status_var.set(f"填充失败: {error_msg}"))
                self.root.after(0, lambda: print(f"自动填充失败: {error_msg}"))
        
        threading.Thread(target=fill_thread, daemon=True).start()
    
    def display_grid(self, grid, title):
        """在终端显示网格"""
        print(f"\n{title}")
        print("=" * 25)
        
        for i in range(9):
            if i % 3 == 0 and i != 0:
                print("------+-------+------")
            
            line = ""
            for j in range(9):
                if j % 3 == 0 and j != 0:
                    line += "| "
                line += str(grid[i][j]) if grid[i][j] != 0 else ". "
            
            print(line)
        
        print("=" * 25)
    
    def run(self):
        """运行GUI"""
        try:
            self.root.mainloop()
        finally:
            # 程序结束时打印性能统计
            perf_config.print_stats()
#程序启动
if __name__ == "__main__":
    # 检查依赖
    try:
        import cv2
        import pytesseract
        import pyautogui
        from PIL import Image, ImageGrab
    except ImportError as e:
        print(f"缺少依赖库: {e}")
        print("请安装以下库:")
        print("pip install opencv-python pytesseract pyautogui pillow")
        print("注意: 还需要安装Tesseract OCR引擎")
        exit(1)
    
    # 设置pyautogui安全设置
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0  # 移除默认延迟以实现快速填充
    # 创建GUI实例并加载默认配置
    app = SudokuGUI()
    app.load_selected_config()  # 加载默认配置
    
    # 启动GUI
    app.run()