row = 9
col = 9


def print_board(bo):
    for i in range(len(bo)):
        if i % 3 == 0 and i != 0:
            print("- - - - - - - - - - - - ")

        for j in range(len(bo[0])):
            if j % 3 == 0 and j != 0:
                print(" | ", end="")

            if j == 8:
                print(bo[i][j])
            else:
                print(str(bo[i][j]) + " ", end="")


def find_empty(board):
    for i in range(row):
        for j in range(col):
            if board[i][j] == 0:
                return (i, j)

    return None


def check_validity(board, pos: tuple, value):
    for i in range(col):
        if board[i][pos[1]] == value and pos[0] != i:
            return False

    for i in range(row):
        if board[pos[0]][i] == value and pos[1] != i:
            return False

    box_X = (pos[0] // 3) * 3
    box_Y = (pos[1] // 3) * 3

    for i in range(box_Y, box_Y + 3):
        for j in range(box_X, box_X + 3):
            if board[j][i] == value and pos != (j, i):
                return False

    return True


def solve(board, pbar):
    find = find_empty(board)
    if find is None:
        return True

    pbar.update(1)

    for i in range(1, row + 1):
        if(check_validity(board, find, i)):
            board[find[0]][find[1]] = i

            if solve(board, pbar):
                return True

            board[find[0]][find[1]] = 0

    return False
