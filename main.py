import csv
import heapq
import math
import pickle
import sys
from datetime import datetime
from time import time
import PyPDF2
import tabula
import numpy as np
import pandas as pd
import os
import tkinter as tk
from pdf_builder import PDF
import shutil
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPainter
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow, QPushButton, QSizePolicy

class RoundedButton(QPushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFixedSize(140, 50)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                border-style: solid;
                border-width: 2px;
                border-radius: 25px;
                border-color: #4CAF50;
                color: white;
                font-size: 16px;
                font-weight: bold;
            }

            QPushButton:hover {
                background-color: #3e8e41;
            }

            QPushButton:pressed {
                background-color: #2c662d;
                border-color: #2c662d;
            }
        """)

class MainWindow(QMainWindow):
    def __init__(self, path_of_orders):
        super().__init__()

        # Set the main window properties
        self.setGeometry(300, 200, 800, 600)
        self.setWindowTitle('Order Calculator')
        self.setStyleSheet("background-color: #2d2d2d;")

        # Create a button for loading orders
        self.load_orders_button = RoundedButton('Load Orders', self)
        self.load_orders_button.move(300, 200)
        self.load_orders_button.clicked.connect(self.load_orders)

        # Create a button for calculating
        self.calculate_button = RoundedButton('Calculate', self)
        self.calculate_button.move(300, 300)
        self.calculate_button.clicked.connect(self.calculate)
        self.calculate_button.setEnabled(False)

        # Create a button for loading new map
        self.load_map_button = RoundedButton('Load new map', self)
        self.load_map_button.move(300, 400)
        self.load_map_button.clicked.connect(self.load_map)

        # Initialize an empty list to hold the selected files
        self.selected_files = []

        # Save the path of orders
        self.path_of_orders = path_of_orders

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QColor(0, 0, 0, 50))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(self.rect(), 20, 20)

    def load_orders(self):
        pdf_files_folder = f"{path_of_orders_general}\\PDF_orders"
        rename_orders(pdf_files_folder)

        # restart file list
        self.selected_files = []

        # Open a file dialog to select files
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setDirectory(pdf_files_folder)
        if file_dialog.exec_():
            # Get the file paths of the selected files
            self.selected_files = file_dialog.selectedFiles()
        self.calculate_button.setEnabled(True)

    def load_map(self):
        # Open a file dialog to select files
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setDirectory(self.path_of_orders)
        if file_dialog.exec_():
            # Get the file paths of the selected files
            selected_map = file_dialog.selectedFiles()
            self.create_graph(selected_map)

    def calculate(self):
        # Call the calculate function with the selected files
        calculate(self.path_of_orders, self.selected_files)

    def create_graph(self, path_to_map):
        if len(path_to_map) > 1:
            create_popup("fail! too many files were chosen!\n please choose 1 file only...", "ok")
        if len(path_to_map) == 0:
            create_popup("fail! no files were chosen!\n please choose 1 file only...", "ok")
        # Read the map from CSV file
        map_file = path_to_map[0]
        with open(map_file, 'r') as f:
            reader = csv.reader(f)
            map = list(reader)
        nodes = {}

        for map_row in range(len(map)):
            for map_col in range(len(map[0])):
                if map[map_row][map_col] != '0':
                    nodes[f"{map[map_row][map_col]}"] = (map_row, map_col)

        graph = {}
        for node in nodes:
            neighbors_and_weights = {}
            for secondary_node in nodes:
                if node != secondary_node:
                    neighbors_and_weights[secondary_node] = shortest_route(node, secondary_node, path_to_map)
            graph[node] = neighbors_and_weights

        map_path = f"{map_file}\\map_graph.pkl"
        # human readable
        if os.path.exists(map_path):
            os.remove(map_path)
        with open(map_file, 'wb') as handle:
            pickle.dump(graph, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("new map saved!")

def create_popup(string1, string2):
    # Define the color palette
    colors = {
        'green': {'bg': '#71cc63', 'fg': 'white'},
        'red': {'bg': '#f44336', 'fg': 'white'},
        'yellow': {'bg': '#ffcc00', 'fg': 'black'},
        'blue': {'bg': '#2196f3', 'fg': 'white'},
        'purple': {'bg': '#9c27b0', 'fg': 'white'},
        'teal': {'bg': '#009688', 'fg': 'white'}
    }

    # Create the pop-up window
    popup = tk.Tk()

    # Determine the background color and icon based on the contents of string1
    if 'success' in string1.lower():
        popup.title('Success!')

        bg_color = colors['green']['bg']
        fg_color = colors['green']['fg']
        icon = '✓'
    elif 'fail' in string1.lower():
        popup.title('Failed!')

        bg_color = colors['red']['bg']
        fg_color = colors['red']['fg']
        icon = '✗'
    else:
        popup.title('Attention!')

        bg_color = colors['blue']['bg']
        fg_color = colors['blue']['fg']
        icon = ''



    popup.configure(bg=bg_color)

    # Add a label with string1 and the icon
    label1 = tk.Label(popup, text=string1, font=('Roboto', 14), bg=bg_color)
    label1.pack(pady=1)
    label1.update()  # Update the label to ensure winfo_reqwidth() and winfo_reqheight() return accurate values
    width = label1.winfo_reqwidth() + 20  # Add a padding of 20 pixels on each side
    height = label1.winfo_reqheight() + 220  # Add extra height for the icon and button

    popup.geometry('{}x{}'.format(width, height))

    icon_label = tk.Label(popup, text=icon, font=('Arial', 30), bg=bg_color, fg=fg_color, activebackground=bg_color, activeforeground=fg_color, borderwidth=0, relief='groove')
    icon_label.pack()

    # Add a button with string2 as its label that closes the window when clicked
    button = tk.Button(popup, text=string2, font=('Roboto', 16), command=popup.destroy, bg=bg_color, fg=fg_color,
                       activebackground=bg_color, activeforeground=fg_color, borderwidth=0, relief='flat', bd=0,
                       highlightthickness=0, highlightbackground=bg_color, highlightcolor=bg_color, pady=10, padx=15)
    button.pack()

    # Configure the button to have rounded corners
    button.configure(
        highlightthickness=0,
        borderwidth=0,
        relief="flat",
        pady=10,
        padx=15,
        background=bg_color,
        foreground=fg_color,
        font=("Roboto", 16),
        command=popup.destroy,
        bd=0,
        highlightbackground=bg_color,
        highlightcolor=bg_color,
        takefocus=False,
    )
    button.configure(
        default=tk.ACTIVE,
        activebackground=bg_color,
        activeforeground=fg_color,
        disabledforeground=fg_color,
        highlightthickness=0,
        pady=10,
        padx=15,
        relief="flat",
        overrelief="flat",
    )
    button.bind("<Enter>", lambda e: button.config(relief="flat"))
    button.bind("<Leave>", lambda e: button.config(relief="flat"))

    # Start the mainloop to display the window
    popup.mainloop()

def rename_orders(path):
    list_of_pdfs = os.listdir(path)
    for file in list_of_pdfs:
        # Open the PDF file and create a PDF reader object
        pdf_file = open(f"{path}\\{file}", 'rb')
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        page = pdf_reader.pages[0]
        bottom_half_of_info = page.extract_text().split('שם')[1]
        customer_name = bottom_half_of_info.split('\n')[0][:-2]
        end_of_file_name = file.split("\\")[-1]
        new_file_name = file.replace(end_of_file_name, customer_name)
        new_file_name = f"{new_file_name}.pdf"
        new_file_name = new_file_name.replace('"', "")

        pdf_file.close()

        os.rename(f"{path}\\{file}", f"{path}\\{new_file_name}")

def calculate(path_of_orders, list_of_files):
    with open(f"{path_of_orders}\\map_graph.pkl", 'rb') as handle:
        map_graph = pickle.load(handle)

    places, df = get_places(list_of_files)
    total_st_time = time()
    places = list(places)
    n = len(places)

    if n==0:
        print("there are no places listed in these orders! please load new orders...")
        create_popup("failed to calculate! there are no places listed in these orders. please load new orders...", "ok")
        return
    subgraph = {place: {} for place in places}

    for place1 in places:
        for place2, dist in map_graph[place1].items():
            if place2 in places:
                subgraph[place1][place2] = dist

    dp = [[math.inf] * (1 << n) for _ in range(n)]
    dp[0][1] = 0
    parent = [[None] * (1 << n) for _ in range(n)]

    for mask in range(1, 1 << n):
        for i in range(n):
            if not (mask & (1 << i)):
                continue
            prev_mask = mask ^ (1 << i)
            for j in range(n):
                if not (prev_mask & (1 << j)):
                    continue
                if dp[j][prev_mask] + subgraph[places[j]][places[i]] < dp[i][mask]:
                    dp[i][mask] = dp[j][prev_mask] + subgraph[places[j]][places[i]]
                    parent[i][mask] = j

    # Find the shortest tour length
    tour_len = math.inf
    last = None
    for i in range(1, n):
        if dp[i][(1 << n) - 1] + subgraph[places[i]][places[0]] < tour_len:
            tour_len = dp[i][(1 << n) - 1] + subgraph[places[i]][places[0]]
            last = i

    # Reconstruct the shortest tour
    tour = [places[0], places[last]]
    mask = (1 << n) - 1
    while parent[last][mask] is not None:
        prev_last = last
        last = parent[last][mask]
        mask ^= (1 << prev_last)
        tour.append(places[last])
    tour.append(places[0])
    tour.reverse()


    print(f"best route: {tour}, total distance: {tour_len}")
    print(f"time of calculation: {time() - total_st_time}")

    # move files from main folder
    single_orders_folder = f"{path_of_orders}\\single_orders"
    os.makedirs(single_orders_folder, exist_ok=True)

    for file in range(len(list_of_files)):
        shutil.move(f"{path_of_orders}\\PDF_orders\\{file}.pdf", f"{single_orders_folder}\\{file}.pdf")
    rename_orders(single_orders_folder)
    build_new_pdf_structure(df, tour, list_of_files)

def get_places(list_of_pdfs):
    df, combined_places = [], []

    for i,file in enumerate(list_of_pdfs):
        new_file_name = f"{path_of_orders_general}\\PDF_orders\\{i}.pdf"

        os.rename(file, new_file_name)
        # Open the PDF file and create a PDF reader object
        pdf_file = open(new_file_name, 'rb')
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        # Get the number of pages in the PDF file
        num_pages = len(pdf_reader.pages)

        # Loop through each page in the PDF file
        for page_num in range(num_pages):
            page = pdf_reader.pages[0]
            customer_sn = page.extract_text().split('לכבוד')[1]
            customer_sn = customer_sn.split('\n')[0][1:-1]
            # Use tabula-py to extract the table from the current page
            tables_in_file = tabula.read_pdf(new_file_name, pages=page_num + 1, encoding='cp1255')
            data_table = max(tables_in_file, key=lambda x: x.size)
            data_table["customer_sn"] = customer_sn
            df.append(data_table)

            # oreder_num = tabula.read_pdf(file, pages=page_num + 1, encoding='cp1252')[7]

            # Get the specific column you want to extract from the table
            letter_column = data_table.iloc[:,0]
            place_by_letter = []
            for place in list(letter_column):
                try:
                    if ~np.isnan(place):
                        place_by_letter.append(place.split("-")[0])
                except:
                    if place is not np.nan:
                        place_by_letter.append(place.split("-")[0])

                # if "-" in place:
                #     place_by_letter.append(place.split(""))
            combined_places.extend(place_by_letter)
            places = set(combined_places)

        # Close the PDF file
        pdf_file.close()

        # print(f"place_by_letter:{place_by_letter}")
        print(f"all places:{places}")
        # print(f"all_places:{combined_places}")
    return places, df

def shortest_route(start_point, end_point, path_to_map):
    map_file = path_to_map[0]
    # Read the map from CSV file
    with open(map_file, 'r') as f:
        reader = csv.reader(f)
        map_data = list(reader)

    # Find the indices of start and end points
    start_index = None
    end_index = None
    for i in range(len(map_data)):
        for j in range(len(map_data[0])):
            if start_point in map_data[i][j]:
                start_index = (i, j)
            if end_point in map_data[i][j]:
                end_index = (i, j)
            if start_index is not None and end_index is not None:
                break
        if start_index is not None and end_index is not None:
            break

    # Initialize the distances to all points as infinite except the start point which is 0
    distances = {(i, j): float('inf') for i in range(len(map_data)) for j in range(len(map_data[0]))}
    distances[start_index] = 0

    # Use a priority queue to keep track of the next nodes to visit
    heap = [(0, start_index)]

    # Initialize a dictionary to keep track of the previous node in the shortest path to each node
    previous = {i: None for i in range(len(map_data) * len(map_data[0]))}

    while heap:
        # Get the node with the smallest distance so far
        (distance, node) = heapq.heappop(heap)

        # Check if we have already found the shortest path to the end point
        if node == end_index:
            break

        # Update the distances to the neighboring nodes
        (i, j) = node
        for (ni, nj) in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
            # if ni >= 0 and ni < len(map_data) and nj >= 0 and nj < len(map_data[0]) and (map_data[ni][nj] == '0' or end_point in map_data[ni][nj]):
            if ni >= 0 and ni < len(map_data) and nj >= 0 and nj < len(map_data[0]):
                new_distance = distance + 1
                if new_distance < distances[(ni, nj)]:
                    distances[(ni, nj)] = new_distance
                    heapq.heappush(heap, (new_distance, (ni, nj)))
                    # previous[ni * len(map_data[0]) + nj] = i * len(map_data[0]) + j

    # If we haven't found a path to the end point, return None
    if distances[end_index] == float('inf'):
        return None

    # Trace the path from end point to start point
    # path = []
    # node = end_index
    # while node is not None:
    #     path.append(map_data[node[0]][node[1]])
    #     node = (previous[node[0] * len(map_data[0]) + node[1]] // len(map_data[0]),
    #             previous[node[0] * len(map_data[0]) + node[1]] % len(map_data[0]))
    # path.reverse()

    return distances[end_index]

def build_new_pdf_structure(df, route, path_of_orders):
    if len(df) >= 1:
        for i in range(len(df)-1):
            df[0] = df[0].append(df[i+1])
    new_df = df[0]
    pdf_columns = ["place", "# in stock", "unit_size", "serial_number", "unit", "to collect ", "on_box:"]
    final_combined_order = pd.DataFrame(columns= new_df.columns)
    for place in route:
        for order_line in new_df.iloc[:,0]:
            try:
                if ~np.isnan(order_line):
                    if place in order_line:
                        line_data = new_df[new_df.iloc[:, 0] == order_line]
                        final_combined_order = pd.concat([final_combined_order, line_data], axis=0, ignore_index=True)
            except:
                if order_line is not np.nan:
                    if place in order_line:
                        line_data = new_df[new_df.iloc[:,0]==order_line]
                        final_combined_order = pd.concat([final_combined_order, line_data], axis=0, ignore_index=True)
    final_combined_order = final_combined_order.rename(columns=dict(zip(final_combined_order.columns, pdf_columns)))
    final_combined_order = final_combined_order.drop_duplicates()
    final_combined_order = final_combined_order.iloc[: , :8]

    # dataframe_to_pdf(final_combined_order, path_of_orders)
    fill_and_save_pdf(final_combined_order, route, path_of_orders)
    return final_combined_order
        # final_combined_order.append(new_df.loc[new_df[:,0][0]== place])

def fill_and_save_pdf(df, route, path):
    p = path[0].split("/")
    combined_orders_output_path = ""
    for i in range(len(p) - 2):
        combined_orders_output_path += f"{p[i]}\\"
    # combined_orders_output_path = Path(path).parent
    combined_pdfs_path = f"{combined_orders_output_path}combined_pdfs"
    os.makedirs(combined_pdfs_path, exist_ok=True)

    cap_time = datetime.now()
    cap_time = cap_time.strftime('%Y-%m-%d_%H-%M-%S')
    # cap_time = cap_time.strftime('%H:%M:%S-%d/%m/%Y')

    pdf = PDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 24)
    pdf.cell(w=0, h=20, txt="combined orders", ln=2, align='C')
    pdf.set_font('Arial', '', 16)
    pdf.multi_cell(w=0, h=20, txt=f"Best route:{route}", align='C')

    pdf.set_font('Arial', '', 14)
    pdf.cell(w=0, h=20, txt=f"created at:{cap_time}", ln=2, align='C')

    cols = df.columns
    # Table Header
    for col in range(len(cols)):
        if col == 3:
            wi = 30
        else:
            wi = 23
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(w=wi, h=10, txt=cols[col], border=1, ln=int(col/(len(cols)-1)), align='C')
    # Table contents
    pdf.set_font('Arial', '', 10)
    for i in range(df.shape[0]):
        for col in range(len(cols)):
            if col == 3:
                wi = 30
            else:
                wi = 23
            ln = int(col / (len(cols) - 1))
            cell_data = df[cols[col]].iloc[i]
            if isinstance(cell_data, str):
                if "?" in cell_data or cell_data is np.nan:
                    cell_data = "--"
                pdf.cell(w=wi, h=10, txt=cell_data, border=1, ln=ln, align='C')
            elif isinstance(cell_data, (int, float)):
                pdf.cell(w=wi, h=10, txt=str(cell_data), border=1, ln=ln, align='C')

            else:
                pdf.cell(w=wi, h=10, txt=cell_data.astype(str), border=1, ln=ln, align='C')

        # pdf.cell(w=40, h=10,
        #          txt=df['feature 2'].iloc[i].astype(str),
        #          border=1, ln=1, align='C')
    pdf.output(f'{combined_pdfs_path}/combined_orders_{cap_time}.pdf', 'F')
    print(f"new PDF of combined orders saved at {combined_pdfs_path}\\combined_orders_{cap_time}.pdf")

    create_popup(f"calculation was successful! new combined PDF saved at {combined_pdfs_path}\\combined_orders_{cap_time}.pdf", 'OK')

def main(path_of_orders_general):
    app = QApplication([])
    window = MainWindow(path_of_orders_general)
    window.show()
    app.exec_()

if __name__ == '__main__':

    # Load the map from an Excel file
    path_of_orders_general = "C:\WH_snake_files"

    if len(sys.argv) > 1:
        path_of_orders_general = sys.argv[1]

    main(path_of_orders_general)

