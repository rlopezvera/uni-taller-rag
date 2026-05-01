"""
Genera vouchers financieros ficticios con PIL.
Ejecutar una vez: python generate_vouchers.py
Requiere: pip install Pillow
"""
from PIL import Image, ImageDraw, ImageFont
import os
import textwrap

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def make_voucher(filename, lines, title="COMPROBANTE DE OPERACION", color=(0, 102, 204)):
    width, height = 480, 520
    img = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(img)

    # Header band
    draw.rectangle([0, 0, width, 70], fill=color)
    draw.text((width // 2, 20), title, fill="white", anchor="mm")
    draw.text((width // 2, 50), "www.banca-ficticia.com.pe", fill=(200, 230, 255), anchor="mm")

    # Body lines
    y = 90
    for key, value in lines:
        draw.text((20, y), key, fill=(100, 100, 100))
        draw.text((460, y), value, fill=(20, 20, 20), anchor="ra")
        draw.line([20, y + 18, 460, y + 18], fill=(220, 220, 220), width=1)
        y += 35

    # Footer
    draw.rectangle([0, height - 50, width, height], fill=(245, 245, 245))
    draw.text((width // 2, height - 25), "Operacion realizada exitosamente", fill=(80, 80, 80), anchor="mm")

    out_path = os.path.join(OUTPUT_DIR, filename)
    img.save(out_path)
    print(f"Generado: {out_path}")


# Voucher 1: Yape P2P
make_voucher(
    "voucher_yape_001.png",
    title="YAPE - TRANSFERENCIA",
    color=(102, 45, 145),
    lines=[
        ("Fecha", "02/05/2024  09:14:32"),
        ("Operacion N°", "YP-20240502-00341"),
        ("Tipo", "Transferencia P2P"),
        ("Monto", "S/  350.00"),
        ("De", "Luis Quispe Mamani"),
        ("Para", "Ana Torres Rojas"),
        ("Celular destino", "9XX-XXX-821"),
        ("Concepto", "Pago cuota depa"),
        ("Estado", "EXITOSO"),
        ("Saldo disponible", "S/  1,240.50"),
    ],
)

# Voucher 2: BCP transferencia
make_voucher(
    "voucher_bcp_transferencia_002.png",
    title="BCP - TRANSFERENCIA INTERBANCARIA",
    color=(0, 60, 120),
    lines=[
        ("Fecha", "15/04/2024  14:22:01"),
        ("N° Operacion", "BCP-INT-20240415-7821"),
        ("Tipo", "Transferencia interbancaria"),
        ("Monto", "S/  4,500.00"),
        ("Moneda", "Soles (PEN)"),
        ("Cuenta origen", "193-XXXXXXXX-0-38"),
        ("Banco destino", "BBVA Peru"),
        ("CCI destino", "011-XXXXXXXXXX-XX"),
        ("Titular destino", "Empresa Comercial SAC"),
        ("Concepto", "Pago factura F001-3412"),
        ("Comision", "S/  0.00"),
        ("Estado", "PROCESADO"),
    ],
)

# Voucher 3: BBVA transferencia internacional
make_voucher(
    "voucher_bbva_internacional_003.png",
    title="BBVA - SWIFT INTERNACIONAL",
    color=(0, 40, 130),
    lines=[
        ("Fecha", "28/04/2024  11:05:44"),
        ("Ref. SWIFT", "BSAB20240428PE00192"),
        ("Tipo", "Transferencia internacional"),
        ("Monto", "USD  15,000.00"),
        ("Equivalente PEN", "S/  56,250.00"),
        ("T/C aplicado", "3.750"),
        ("Cuenta origen", "0011-XXXXXXXX-00"),
        ("Banco beneficiario", "Banco Santander Uruguay"),
        ("SWIFT beneficiario", "BSURUS33"),
        ("Beneficiario", "Carlos Mendoza EIRL"),
        ("Proposito", "Pago proveedor materia prima"),
        ("Estado", "EN PROCESO"),
    ],
)

# Voucher 4: Interbank deposito
make_voucher(
    "voucher_interbank_deposito_004.png",
    title="INTERBANK - DEPOSITO EN VENTANILLA",
    color=(0, 156, 68),
    lines=[
        ("Fecha", "03/05/2024  10:30:15"),
        ("N° Operacion", "IB-DEP-20240503-0091"),
        ("Tipo", "Deposito en efectivo"),
        ("Monto", "S/  12,800.00"),
        ("Moneda", "Soles (PEN)"),
        ("Cuenta destino", "898-XXXXXXXX-1-XX"),
        ("Titular", "Rodriguez Vasquez, Carmen"),
        ("DNI", "4XXXXXXX"),
        ("Agencia", "La Victoria - Av. Mexico"),
        ("Cajero N°", "IB-0042"),
        ("Concepto declarado", "Venta de bien inmueble"),
        ("Estado", "COMPLETADO"),
    ],
)

# Voucher 5: Scotiabank pago credito
make_voucher(
    "voucher_scotiabank_credito_005.png",
    title="SCOTIABANK - PAGO DE PRESTAMO",
    color=(255, 0, 0),
    lines=[
        ("Fecha", "01/05/2024  08:00:00"),
        ("N° Prestamo", "SB-PRS-2023-004521"),
        ("Tipo", "Cuota mensual"),
        ("Cuota N°", "6 de 24"),
        ("Capital", "S/  380.42"),
        ("Intereses", "S/  70.33"),
        ("Total cuota", "S/  450.75"),
        ("Saldo capital", "S/  9,880.50"),
        ("Forma de pago", "Debito automatico"),
        ("Cuenta debito", "004-XXXXXXXXXX"),
        ("Proximo vencimiento", "01/06/2024"),
        ("Estado", "PAGADO"),
    ],
)

print("\nTodos los vouchers generados exitosamente.")
