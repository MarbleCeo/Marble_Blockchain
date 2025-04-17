import sys
import threading
import time
from PyQt5 import QtWidgets, QtCore, QtGui
from p2p_node import P2PNode

class GUI(QtWidgets.QMainWindow):
    def __init__(self, node):
        super().__init__()
        self.node = node

        self.setWindowTitle('P2P Blockchain Chat')
        self.setGeometry(100, 100, 800, 600)

        self.nickname = None
        self.chat_history = QtWidgets.QTextEdit(self)
        self.chat_history.setReadOnly(True)

        self.entry_nickname = QtWidgets.QLineEdit(self)
        self.button_set_nickname = QtWidgets.QPushButton('Set Nickname', self)
        self.button_set_nickname.clicked.connect(self.set_nickname)

        self.entry_message = QtWidgets.QLineEdit(self)
        self.button_send = QtWidgets.QPushButton('Send', self)
        self.button_send.clicked.connect(self.send_message)

        self.entry_transaction = QtWidgets.QLineEdit(self)
        self.button_add_transaction = QtWidgets.QPushButton('Add Transaction', self)
        self.button_add_transaction.clicked.connect(self.add_transaction)

        central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(central_widget)

        layout = QtWidgets.QGridLayout(central_widget)
        layout.addWidget(self.chat_history, 0, 0, 4, 2)
        layout.addWidget(self.entry_nickname, 4, 0)
        layout.addWidget(self.button_set_nickname, 4, 1)
        layout.addWidget(self.entry_message, 5, 0)
        layout.addWidget(self.button_send, 5, 1)
        layout.addWidget(self.entry_transaction, 6, 0)
        layout.addWidget(self.button_add_transaction, 6, 1)

        self.update_chat_history_thread = threading.Thread(target=self.update_chat_history)
        self.update_chat_history_thread.start()

    def run(self):
        self.show()
        sys.exit(QtWidgets.QApplication.exec_())

    def set_nickname(self):
        self.nickname = self.entry_nickname.text()

    def send_message(self):
        if not self.nickname:
            print('Please set a nickname first.')
            return

        message = self.entry_message.text()
        self.node.send_message(self.nickname, message)
        self.entry_message.clear()

    def add_transaction(self):
        transaction = self.entry_transaction.text()
        self.node.add_transaction(transaction)
        self.entry_transaction.clear()

    def update_chat_history(self):
        while True:
            chat_history = self.node.get_chat_history()
            self.chat_history.setPlainText(chat_history)
            time.sleep(1)

    def on_closing(self):
        self.node.stop()
        self.close()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    node = P2PNode('127.0.0.1', 55555)
    node.start()

    gui = GUI(node)
    gui.run()