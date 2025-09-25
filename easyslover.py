# 数独求解器：用回溯法解决数独问题
# 核心思路：尝试填数→检查合法性→递归深入→错了就回溯


def is_valid_number(board, row, col, number):
    """
    检查在指定位置填入数字是否合法
    参数：
        board: 数独棋盘（9x9列表，0表示空格）
        row: 行索引（0-8）
        col: 列索引（0-8）
        number: 要填入的数字（1-9）
    返回：
        True：合法（可填入）；False：不合法（重复）
    """
    # 1. 检查当前行是否有重复数字
    for col_index in range(9):  # 遍历当前行的每一列
        if board[row][col_index] == number:
            return False  # 行里有重复，不合法
    
    # 2. 检查当前列是否有重复数字
    for row_index in range(9):  # 遍历当前列的每一行
        if board[row_index][col] == number:
            return False  # 列里有重复，不合法
    
    # 3. 检查当前3x3小宫是否有重复数字
    # 计算小宫的左上角坐标（每个小宫是3x3的区域）
    start_row_of_box = (row // 3) * 3  # 小宫起始行（0, 3, 6）
    start_col_of_box = (col // 3) * 3  # 小宫起始列（0, 3, 6）
    
    # 遍历小宫的3行3列
    for i in range(start_row_of_box, start_row_of_box + 3):
        for j in range(start_col_of_box, start_col_of_box + 3):
            if board[i][j] == number:
                return False  # 小宫里有重复，不合法
    
    # 所有检查通过，合法
    return True


def solve_sudoku_puzzle(board):
    """
    用回溯法求解数独
    参数：
        board: 数独棋盘（9x9列表，0表示空格）
    返回：
        True：找到解；False：无解
    """
    # 1. 遍历整个数独，找到第一个空格（值为0的位置）
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:  # 找到空格
                
                # 2. 尝试在这个空格填入1-9的每个数字
                for number in range(1, 10):
                    # 先检查这个数字是否合法
                    if is_valid_number(board, row, col, number):
                        # 合法就填入数字
                        board[row][col] = number
                        
                        # 3. 递归求解剩下的数独（填好当前数后，继续解后面的空格）
                        if solve_sudoku_puzzle(board):
                            return True  # 如果递归成功，说明整个数独解出来了
                        
                        # 4. 回溯：如果递归失败，说明当前数字填错了，恢复空格重新尝试
                        board[row][col] = 0
                
                # 5. 如果1-9都试了还不行，说明这个空格无解，返回失败
                return False
    
    # 6. 遍历完所有格子，没有空格了→数独已解，返回成功
    return True


def print_sudoku_board(board):
    """
    美观地打印数独棋盘，用分隔线区分3x3小宫
    参数：
        board: 数独棋盘（9x9列表）
    """
    # 打印顶部边框
    print("┌───────┬───────┬───────┐")
    
    for row in range(9):
        # 打印当前行内容（用│分隔小宫）
        print("│ ", end="")  # 行首边框
        for col in range(9):
            # 用"·"代替0表示空格，更直观
            cell = board[row][col] if board[row][col] != 0 else "·"
            print(f"{cell} ", end="")
            # 每3列加一个竖线分隔小宫
            if (col + 1) % 3 == 0:
                print("│ ", end="")
        print()  # 换行
        
        # 每3行加一个横线分隔小宫（最后一行不加）
        if (row + 1) % 3 == 0 and row != 8:
            print("├───────┼───────┼───────┤")
    
    # 打印底部边框
    print("└───────┴───────┴───────┘")


# 主程序：运行数独求解
if __name__ == "__main__":
    # 定义数独题目（0表示空格）
    sudoku_board = [
        [0, 0, 0, 7, 4, 0, 0, 0, 0],
        [7, 0, 0, 0, 9, 5, 3, 0, 0],
        [0, 0, 0, 1, 0, 2, 0, 0, 7],
        [0, 0, 0, 5, 6, 0, 0, 0, 0],
        [9, 0, 0, 0, 0, 0, 7, 0, 3],
        [0, 0, 0, 0, 0, 0, 0, 4, 0],
        [0, 7, 0, 0, 0, 0, 0, 0, 2],
        [0, 0, 2, 3, 0, 9, 0, 0, 0],
        [0, 0, 4, 2, 8, 0, 1, 3, 0]
    ]
    
    # 打印题目
    print("数独题目：")
    print_sudoku_board(sudoku_board)
    
    # 求解并打印结果
    if solve_sudoku_puzzle(sudoku_board):
        print("\n数独答案：")
        print_sudoku_board(sudoku_board)
    else:
        print("\n这个数独没有解！")