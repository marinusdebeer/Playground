import queue

# Initialize the game board
grid = [[0 for _ in range(25)] for _ in range(25)]
snake = [(12, 12), (12, 13), (12, 14)]  # example snake position
for x, y in snake:
    grid[x][y] = 1

# Find the head and tail of the snake
head = snake[-1]
tail = snake[0]

# Use BFS to find a path from the head to the tail
visited = set()
path = {}
q = queue.Queue()
q.put(head)
visited.add(head)
while not q.empty():
    curr = q.get()
    if curr == tail:
        break
    x, y = curr
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < 25 and 0 <= ny < 25 and grid[nx][ny] == 0 and (nx, ny) not in visited:
            q.put((nx, ny))
            visited.add((nx, ny))
            path[(nx, ny)] = path[curr] + [(nx, ny)]

# Move the snake along the path, avoiding collisions with its own body
while head != tail:
    # Find the next cell to move to
    next_cell = path[head][0]
    dx, dy = next_cell[0] - head[0], next_cell[1] - head[1]

    # Update the game board and snake position
    grid[head[0]][head[1]] = 0
    head = (head[0] + dx, head[1] + dy)
    snake.append(head)
    snake.pop(0)
    grid[head[0]][head[1]] = 1

    # Check for collisions with the snake's own body
    if head in snake[:-1]:
        print("Game over!")
        break

# Print the final state of the game board
for row in grid:
    print(" ".join(str(x) for x in row))
