import tkinter
from tkinter import messagebox, scrolledtext
import pickle
from PIL import Image,ImageTk
from GUI_1 import interface
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文
class Login(object):
    def __init__(self):
        # 创建主窗口
        self.root = tkinter.Tk()
        self.root.title('基于图像处理的草莓严重度估算系统---- CAU')
        self.root.geometry('400x300+550+200')
        self.root.resizable(0, 0)

        self.root.canvas = tkinter.Canvas(self.root, width=400, height=135, bg='blue')
        image = Image.open('./1.jpg')#用作背景图
        self.root.image_file = ImageTk .PhotoImage(image)
        self.root.image = self.root.canvas.create_image(200, 0, anchor='n', image=self.root.image_file)
        self.root.canvas.pack(side='top')
        tkinter.Label(self.root, text='欢迎您', font=('Arial', 16)).pack()

        tkinter.Label(self.root, text='用户名:', font=('Arial', 14)).place(x=10, y=170)
        tkinter.Label(self.root, text='密码:', font=('Arial', 14)).place(x=10, y=210)


        # 用户名
        self.root.var_usr_name = tkinter.StringVar()
        self.root.var_usr_name.set("")
        self.root.entry_usr_name =tkinter.Entry(self.root, textvariable=self.root.var_usr_name, font=('Arial', 14))
        self.root.entry_usr_name.place(x=120, y=175)
        # 用户密码
        self.root.var_usr_pwd = tkinter.StringVar()
        self.root.entry_usr_pwd = tkinter.Entry(self.root, textvariable=self.root.var_usr_pwd, font=('Arial', 14), show='*')
        self.root.entry_usr_pwd.place(x=120, y=215)


        def usr_login():
            # 这两行代码就是获取用户输入的usr_name和usr_pwd
            usr_name = self.root.var_usr_name.get()
            usr_pwd = self.root.var_usr_pwd.get()

            try:
                with open('../../usrs_info.pickle', 'rb') as usr_file:
                    usrs_info = pickle.load(usr_file)
            except FileNotFoundError:
                with open('../../usrs_info.pickle', 'wb') as usr_file:
                    usrs_info = {'admin': 'admin'}
                    pickle.dump(usrs_info, usr_file)
                    usr_file.close()

            # 如果用户名和密码与文件中的匹配成功，则会登录成功，并跳出弹窗欢迎登录 加上你的用户名。
            if usr_name in usrs_info:
                if usr_pwd == usrs_info[usr_name]:
                    tkinter.messagebox.showinfo(title='Welcome', message='欢迎登陆！'+usr_name)
                    self.root.destroy()
                    this_ill = interface()
                    this_ill.interface.mainloop()

                else:
                    tkinter.messagebox.showerror(message='你密码错了')
            else:  # 如果发现用户名不存在
                is_sign_up = tkinter.messagebox.askyesno('欢迎您', '用户名不存在')
                # 提示需不需要注册新用户
                if is_sign_up:
                    usr_sign_up()

        # 第9步，定义用户注册功能
        def usr_sign_up():
            def sign_to_file():
                # 以下三行就是获取我们注册时所输入的信息
                np = new_pwd.get()
                npf = new_pwd_confirm.get()
                nn = new_name.get()

                # 这里是打开我们记录数据的文件，将注册信息读出
                with open('../../usrs_info.pickle', 'rb') as usr_file:
                    exist_usr_info = pickle.load(usr_file)

                if np != npf:
                    tkinter.messagebox.showerror('Error', '两次输入密码不一致')

                elif nn in exist_usr_info:
                    tkinter.messagebox.showerror('Error', '用户已注册')

                else:
                    exist_usr_info[nn] = np
                    with open('../../usrs_info.pickle', 'wb') as usr_file:
                        pickle.dump(exist_usr_info, usr_file)
                    tkinter.messagebox.showinfo('Welcome', '注册成功')

                    window_sign_up.destroy()

            # 定义长在窗口上的窗口
            window_sign_up = tkinter.Toplevel(self.root)
            window_sign_up.geometry('300x200')
            window_sign_up.title('Sign up window')

            new_name = tkinter.StringVar()
            new_name.set('')
            tkinter.Label(window_sign_up, text='用户名: ').place(x=10, y=10)
            entry_new_name = tkinter.Entry(window_sign_up, textvariable=new_name)
            entry_new_name.place(x=130, y=10)

            new_pwd = tkinter.StringVar()
            tkinter.Label(window_sign_up, text='密码: ').place(x=10, y=50)
            entry_usr_pwd = tkinter.Entry(window_sign_up, textvariable=new_pwd, show='*')
            entry_usr_pwd.place(x=130, y=50)

            new_pwd_confirm = tkinter.StringVar()
            tkinter.Label(window_sign_up, text='确认密码: ').place(x=10, y=90)
            entry_usr_pwd_confirm = tkinter.Entry(window_sign_up, textvariable=new_pwd_confirm, show='*')
            entry_usr_pwd_confirm.place(x=130, y=90)

            btn_comfirm_sign_up = tkinter.Button(window_sign_up, text='注册', command=sign_to_file)
            btn_comfirm_sign_up.place(x=180, y=120)

        btn_login = tkinter.Button(self.root, text='登录', command=usr_login)
        btn_login.place(x=120, y=240)
        btn_sign_up = tkinter.Button(self.root, text='注册', command=usr_sign_up)
        btn_sign_up.place(x=200, y=240)

if __name__ == '__main__':
    this_main = Login()
    this_main.root.mainloop()