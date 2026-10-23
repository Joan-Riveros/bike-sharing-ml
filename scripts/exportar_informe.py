"""Exporta reports/joan/informe_tecnico.md a PDF (Edge sin interfaz) y a Word (.docx, automatización de Word).

Uso, desde la raíz del repositorio y con el entorno activado:
    python scripts/exportar_informe.py            # PDF y Word
    python scripts/exportar_informe.py --solo-pdf
    python scripts/exportar_informe.py --solo-docx

Solo para Windows con Microsoft Edge y Microsoft Word instalados; no instala nada. El Markdown se convierte a HTML con
``mistune`` (ya viene con ``nbconvert``). Los HTML intermedios se escriben en una carpeta temporal, no en el repositorio.
"""
import argparse
import base64
import mimetypes
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import mistune

AQUI = Path(__file__).resolve().parent
INFORME = AQUI.parent / "reports" / "joan"          # carpeta del informe; las figuras se enlazan desde aquí
MD = INFORME / "informe_tecnico.md"
PDF = INFORME / "informe_tecnico.pdf"
DOCX = INFORME / "informe_tecnico.docx"

CSS = """
@page { size: A4; margin: 17mm 15mm 19mm 15mm; @bottom-center { content: counter(page) " / " counter(pages); font-size: 8pt; color: #666; } }
body { font-family: "Segoe UI", Calibri, Arial, sans-serif; font-size: 9.6pt; line-height: 1.38; color: #1b1b1b; }
h1 { font-size: 17pt; margin: 0 0 6px 0; color: #12355b; }
h2 { font-size: 12.5pt; margin: 15px 0 5px 0; padding-bottom: 2px; border-bottom: 1.5px solid #12355b; color: #12355b; page-break-after: avoid; }
h3 { font-size: 10.6pt; margin: 11px 0 4px 0; color: #1f4e79; page-break-after: avoid; }
p { margin: 4px 0 6px 0; } ul, ol { margin: 3px 0 6px 0; padding-left: 20px; } li { margin: 1.5px 0; }
table { width: 100%; border-collapse: collapse; margin: 6px 0 9px 0; font-size: 8.3pt; }
th { background: #e8eef5; text-align: left; } th, td { border: 1px solid #b9c3cf; padding: 2.5px 5px; vertical-align: top; }
tr { page-break-inside: avoid; }
code { font-family: Consolas, "Courier New", monospace; font-size: 8.6pt; background: #f2f2f2; padding: 0 2px; }
blockquote { margin: 5px 0; padding: 3px 9px; border-left: 3px solid #9db4cc; background: #f5f8fb; color: #333; }
img { display: block; margin: 5px auto; max-width: 100%; max-height: 7.2cm; }
hr { border: 0; border-top: 1px solid #ccd; margin: 8px 0; }
"""


def a_html(incrustar: bool) -> str:
    """Markdown -> HTML. Con ``incrustar`` las imágenes van dentro del HTML (base64); si no, como rutas de archivo."""
    md = mistune.create_markdown(escape=True, plugins=["table", "strikethrough"])
    cuerpo = md(MD.read_text(encoding="utf-8"))

    def imagen(m):
        ruta = (INFORME / m.group(1)).resolve()
        if not ruta.exists():
            raise FileNotFoundError(f"Falta la imagen del informe: {ruta}")
        if incrustar:
            tipo = mimetypes.guess_type(ruta.name)[0] or "image/png"
            return f'src="data:{tipo};base64,{base64.b64encode(ruta.read_bytes()).decode()}"'
        return f'src="{ruta.as_uri()}"'

    cuerpo = re.sub(r'src="([^"]+)"', imagen, cuerpo)
    return f'<!doctype html><html lang="es"><head><meta charset="utf-8"><title>Informe técnico — Bike Sharing Demand</title><style>{CSS}</style></head><body>{cuerpo}</body></html>'


def buscar_edge() -> Path:
    for p in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"):
        if Path(p).exists():
            return Path(p)
    raise FileNotFoundError("No se encontró Microsoft Edge")


def exportar_pdf(carpeta: Path) -> None:
    html = carpeta / "informe_pdf.html"
    html.write_text(a_html(incrustar=True), encoding="utf-8")
    if PDF.exists():
        PDF.unlink()
    subprocess.run([str(buscar_edge()), "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={PDF}", html.as_uri()],
                   check=True, timeout=180, capture_output=True)
    if not PDF.exists() or PDF.stat().st_size < 10_000:
        raise RuntimeError("Edge no generó el PDF")
    print(f"PDF  -> {PDF.relative_to(AQUI.parent)} ({PDF.stat().st_size / 1e6:.2f} MB)")


def exportar_docx(carpeta: Path) -> None:
    html = carpeta / "informe_word.html"
    html.write_text(a_html(incrustar=False), encoding="utf-8")
    if DOCX.exists():
        DOCX.unlink()
    ps = f"""
$ErrorActionPreference = 'Stop'
$w = New-Object -ComObject Word.Application
try {{
  $w.Visible = $false; $w.DisplayAlerts = 0
  $d = $w.Documents.Open('{html}', $false, $true)
  # Las imágenes del HTML se abren como vínculos a archivos: se incrustan para que el .docx sea autónomo
  for ($i = $d.InlineShapes.Count; $i -ge 1; $i--) {{
    $s = $d.InlineShapes.Item($i)
    if ($s.Type -eq 4) {{ $s.LinkFormat.SavePictureWithDocument = $true; $s.LinkFormat.BreakLink() }}
  }}
  # Formato de página: A4 con márgenes de 1.8 cm, tablas al ancho de la página, figuras acotadas y numeración
  $ps = $d.PageSetup
  $ps.PageWidth = 595.3; $ps.PageHeight = 841.9
  $ps.LeftMargin = 51; $ps.RightMargin = 51; $ps.TopMargin = 51; $ps.BottomMargin = 54
  $ancho = $ps.PageWidth - $ps.LeftMargin - $ps.RightMargin
  foreach ($t in $d.Tables) {{ $t.AllowAutoFit = $true; $t.PreferredWidthType = 2; $t.PreferredWidth = 100; $t.Range.ParagraphFormat.SpaceAfter = 0 }}
  foreach ($s in $d.InlineShapes) {{
    if ($s.Type -ne 3) {{ continue }}      # 3 = imagen; se omiten las líneas horizontales (<hr>)
    $s.LockAspectRatio = -1
    if ($s.Width -gt $ancho) {{ $s.Width = $ancho }}
    if ($s.Height -gt 205) {{ $s.Height = 205 }}
  }}
  # Word ignora el CSS de tamaños y usa Times 12: se fijan fuentes y espaciado por estilo
  foreach ($nombre in @('Normal', 'Normal (Web)')) {{
    try {{ $st = $d.Styles.Item($nombre); $st.Font.Name = 'Calibri'; $st.Font.Size = 10; $st.ParagraphFormat.SpaceBefore = 0; $st.ParagraphFormat.SpaceAfter = 4; $st.ParagraphFormat.LineSpacingRule = 0 }} catch {{ }}
  }}
  foreach ($par in @(@(-2, 16), @(-3, 13), @(-4, 11))) {{      # Título 1, 2 y 3
    $st = $d.Styles.Item($par[0]); $st.Font.Name = 'Calibri'; $st.Font.Size = $par[1]; $st.Font.Bold = $true; $st.Font.Color = 0x5B3512
    $st.ParagraphFormat.SpaceBefore = 10; $st.ParagraphFormat.SpaceAfter = 4; $st.ParagraphFormat.KeepWithNext = $true
  }}
  foreach ($t in $d.Tables) {{ $t.Borders.InsideLineStyle = 1; $t.Borders.OutsideLineStyle = 1; $t.Borders.InsideColor = 0xCFC3B9; $t.Borders.OutsideColor = 0xCFC3B9; $t.Range.Font.Name = 'Calibri'; $t.Range.Font.Size = 8.5; $t.Rows.Item(1).Range.Font.Bold = $true; $t.Rows.Item(1).Shading.BackgroundPatternColor = 0xF5EEE8 }}
  $d.Sections.Item(1).Footers.Item(1).PageNumbers.Add(1, $true) | Out-Null
  $d.SaveAs2('{DOCX}', 16)      # 16 = wdFormatXMLDocument (.docx)
  $d.Close($false)
}} finally {{ $w.Quit() }}
"""
    r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps], timeout=300, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError("Word falló:\n" + (r.stderr or r.stdout)[:1500])
    if not DOCX.exists():
        raise RuntimeError("Word no generó el .docx")
    print(f"DOCX -> {DOCX.relative_to(AQUI.parent)} ({DOCX.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--solo-pdf", action="store_true"); ap.add_argument("--solo-docx", action="store_true")
    a = ap.parse_args()
    if sys.platform != "win32":
        sys.exit("Este exportador usa Edge y Word y solo funciona en Windows.")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tmp = Path(tmp)
        if not a.solo_docx:
            exportar_pdf(tmp)
        if not a.solo_pdf:
            exportar_docx(tmp)
