# lay = '''
# X 0 0 0 0 # 0 0 0, 
# 0 0 0 0 0 # # # 0, 
# 0 0 0 0 0 0 X # 0, 
# 0 0 0 0 0 0 0 0 0, 
# 0 0 # # # 0 0 0 0, 
# 0 0 # R 0 0 0 0 X, 
# 0 0 # 0 0 0 0 0 0, 
# 0 0 0 0 0 0 0 0 0, 
# X 0 0 0 0 0 0 0 0
# '''

lay = '''
0 0 0 0 0 0 0 0 0,
X 0 0 0 R 0 0 0 X, 
0 0 0 0 0 0 0 0 0
'''




filedata = '''
(define (problem grid10)

(:domain grid)
(:objects
    <OBJECTS-HERE>
)
(:init
    <INIT-HERE>
)

(:goal
(and 
      <HYPOTHESIS>
)
))
'''




max_r = len(lay.split(','))
max_c = len("".join(lay.split(',')[0].split()))


conn = []
walls =[]
goals = []
objects = []


for r in range(max_r):
    for c in range(max_c):
        place = f'place_{r}_{c}'
        objects.append(f'{place} - place')
        if c+1 < max_c:
            conn.append(f'(conn {place} place_{r}_{c+1})')
        if c-1 >= 0:
            conn.append(f'(conn {place} place_{r}_{c-1})')
        if r+1 < max_r:
            conn.append(f'(conn {place} place_{r+1}_{c})')
        if r-1 >= 0:
            conn.append(f'(conn {place} place_{r-1}_{c})')

        x = "".join(lay.split(',')[r].split())[c]
        if x == '#':
            walls.append(f'(wall place_{r}_{c})')
        if x == 'R':
            robot = f'(at-robot place_{r}_{c})'
        if x == 'X':
            goals.append(f'(is-goal place_{r}_{c})')


conn = list(set(conn)) 

init = []
init += conn
init += walls
init += goals
init.append(robot)


for x in objects:
    print(x)

for x in init:
    print(x)





# with open('template.pddl', 'w') as file:
#     filedata = filedata.replace('<OBJECTS-HERE>', objects)
#     filedata = filedata.replace('<INIT-HERE>', init)
#     file.write(filedata)




# walls = []
# goals = []

# for r, x in enumerate(lay.split(',')):
#     for c, y in enumerate("".join(x.split())):
#         if y == '1':
#             walls.append(f'(wall place_{r}_{c})')
#         if y == 'R':
#             robot = f'(at-robot place_{r}_{c})'
#         if y == 'X':
#             goals.append(f'(is-goal place_{r}_{c})')


# objects = []
# objects += walls
# objects += goals
# objects.append(robot)

# for x in objects:
#     print(x)


print(0.24999999999999986/(0.062499999999999965 + 0.24999999999999986 + 0.062499999999999965))

print(0.24999999999999986/(0.062499999999999965 + 0.24999999999999986 + 0.062499999999999965))