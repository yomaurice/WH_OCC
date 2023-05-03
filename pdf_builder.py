from fpdf import FPDF


class PDF(FPDF):
    def __init__(self):
        super().__init__()

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', '', 12)
        self.cell(0, 8, f'Page {self.page_no()}', 0, 0, 'C')