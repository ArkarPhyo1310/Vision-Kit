import subprocess
from typing import Iterable, Optional

from rich import box
from rich.console import Console, JustifyMethod
from rich.padding import PaddingDimensions
from rich.style import StyleType
from rich.table import Table
from rich.text import Text
from vision_kit.utils.logging_utils import logger

console = Console()


class RichTable:
    def __init__(self, title=None, caption=None, width=None, min_width=None) -> None:
        super().__init__()
        self.rtable = Table(title=title, caption=caption,
                            width=width, min_width=min_width)
        self.set_format()
        self.set_style()
        self.show_options()

    def show_box_styles(self):
        subprocess.run("python -m rich.box", shell=True, encoding="utf-8")

    def set_format(
        self,
        width: Optional[int] = None,
        min_width: Optional[int] = None,
        box: Optional[box.Box] = box.HEAVY_HEAD,
        safe_box: Optional[bool] = None,
        padding: PaddingDimensions = (0, 1),
        collapse_padding: bool = False,
        pad_edge: bool = True,
        expand: bool = False,
    ) -> None:
        """Rich Table Formatting

        Args:
            width (int, optional): The width in characters of the table, or ``None`` to automatically fit. Defaults to None.
            min_width (Optional[int], optional): The minimum width of the table, or ``None`` for no minimum. Defaults to None.
            box (box.Box, optional): One of the constants in box.py used to draw the edges (see :ref:`appendix_box`), or ``None`` for no box lines. Defaults to box.HEAVY_HEAD.
            safe_box (Optional[bool], optional): Disable box characters that don't display on windows legacy terminal with *raster* fonts. Defaults to True.
            padding (PaddingDimensions, optional): Padding for cells (top, right, bottom, left). Defaults to (0, 1).
            collapse_padding (bool, optional): Enable collapsing of padding around cells. Defaults to False.
            pad_edge (bool, optional): Enable padding of edge cells. Defaults to True.
            expand (bool, optional): Expand the table to fit the available space if ``True``, otherwise the table width will be auto-calculated. Defaults to False.
        """
        self.rtable.width = width
        self.rtable.min_width = min_width
        self.rtable.box = box
        self.rtable.safe_box = safe_box
        self.rtable.grid(
            padding=padding,
            collapse_padding=collapse_padding,
            pad_edge=pad_edge,
            expand=expand
        )

    def set_style(
        self,
        style: StyleType = "none",
        row_styles: Optional[Iterable[StyleType]] = None,
        header_style: Optional[StyleType] = "table.header",
        footer_style: Optional[StyleType] = "table.footer",
        border_style: Optional[StyleType] = None,
        title_style: Optional[StyleType] = None,
        caption_style: Optional[StyleType] = None,
        title_justify: "JustifyMethod" = "center",
        caption_justify: "JustifyMethod" = "center",
        highlight: bool = False,
    ) -> None:
        """Rich Table Style Setting

        Args:
            style (Union[str, Style], optional): Default style for the table. Defaults to "none".
            row_styles (List[Union, str], optional): Optional list of row styles, if more than one style is given then the styles will alternate. Defaults to None.
            header_style (Union[str, Style], optional): Style of the header. Defaults to "table.header".
            footer_style (Union[str, Style], optional): Style of the footer. Defaults to "table.footer".
            border_style (Union[str, Style], optional): Style of the border. Defaults to None.
            title_style (Union[str, Style], optional): Style of the title. Defaults to None.
            caption_style (Union[str, Style], optional): Style of the caption. Defaults to None.
            title_justify (str, optional): Justify method for title. Defaults to "center".
            caption_justify (str, optional): Justify method for caption. Defaults to "center".
            highlight (bool, optional): Highlight cell contents (if str). Defaults to False.
        """
        self.rtable.style = style
        self.rtable.row_styles = row_styles
        self.rtable.header_style = header_style
        self.rtable.footer_style = footer_style
        self.rtable.border_style = border_style
        self.rtable.title_style = title_style
        self.rtable.caption_style = caption_style
        self.rtable.title_justify = title_justify
        self.rtable.caption_justify = caption_justify
        self.rtable.highlight = highlight

    def show_options(
        self,
        show_header: bool = True,
        show_footer: bool = False,
        show_edge: bool = True,
        show_lines: bool = False,
    ) -> None:
        """Show Table Header, Footer, Edge and Lines

        Args:
            show_header (bool, optional): Show a header row. Defaults to True.
            show_footer (bool, optional): Show a footer row. Defaults to False.
            show_edge (bool, optional): Draw a box around the outside of the table. Defaults to True.
            show_lines (bool, optional): Draw lines between every row. Defaults to False.
        """
        self.rtable.show_header = show_header
        self.rtable.show_footer = show_footer
        self.rtable.show_edge = show_edge
        self.rtable.show_lines = show_lines

    def add_headers(
        self,
        headers: list,
        justify: str = "right",
        style: str = "cyan",
        no_warp: bool = True
    ) -> None:
        headers = [headers] if isinstance(headers, str) else headers

        for header in headers:
            self.rtable.add_column(header, justify=justify,
                                   style=style, no_wrap=no_warp)

    def add_content(self, data: list) -> None:
        if not isinstance(data, list):
            logger.error(
                f"Content must be 2-D list. [['a', 'b'], ['c', 'd']]", exc_info=1)
            # print(
            #     f"Content must be 2-D list. [['a', 'b'], ['c', 'd']]")
        data = data if isinstance(data[0], list) else [data]

        for row in data:
            string_row = list(map(str, row))
            self.rtable.add_row(*string_row)

    @property
    def table(self) -> Text:
        """Generate an ascii formatted presentation of a Rich table
        Eliminates any column styling
        """
        with console.capture() as capture:
            console.print(self.rtable)
        return Text.from_ansi(capture.get())
