import subprocess

# 定义两个程序的路径
program1 = r"D:\yan\Mitsuba3\Render_Mitsuba3\9dataprocess_ng_right.py"
program2 = r"D:\yan\Mitsuba3\Render_Mitsuba3\9dataprocess_gn.py"
program3 = r"D:\yan\Mitsuba3\Render_Mitsuba3\9dataprocess_ng_Stokes_origin.py"

# 使用subprocess.Popen来启动两个程序
process1 = subprocess.Popen(["python", program1])
process2 = subprocess.Popen(["python", program2])
process3 = subprocess.Popen(["python", program3])

# 等待两个程序运行结束
process1.wait()
process2.wait()
process3.wait()

print("两个程序都已运行完毕。")
