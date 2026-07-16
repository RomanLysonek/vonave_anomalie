"""Small flat-Parquet reader for this repository's bundled data files.

Used only in restricted environments without pyarrow. Supports the exact
primitive/Snappy/data-page-v1 layout emitted for train_data.parquet and
test_data.parquet.
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from thrift.Thrift import TType
from thrift.protocol.TCompactProtocol import TCompactProtocol
from thrift.transport.TTransport import TMemoryBuffer

BOOLEAN, INT32, INT64, FLOAT, DOUBLE, BYTE_ARRAY = 0, 1, 2, 4, 5, 6
PLAIN, RLE_DICTIONARY = 0, 8
UNCOMPRESSED, SNAPPY = 0, 1
DATA_PAGE, DICTIONARY_PAGE = 0, 2


def _list(p: TCompactProtocol, reader: Callable | None = None) -> list[Any]:
    etype, size = p.readListBegin()
    out = []
    for _ in range(size):
        if reader:
            out.append(reader(p))
        elif etype == TType.STRING:
            out.append(p.readString())
        elif etype == TType.I32:
            out.append(p.readI32())
        elif etype == TType.I64:
            out.append(p.readI64())
        else:
            p.skip(etype)
            out.append(None)
    p.readListEnd()
    return out


def _schema(p: TCompactProtocol) -> dict[str, Any]:
    d = {}
    p.readStructBegin()
    while True:
        _, t, fid = p.readFieldBegin()
        if t == TType.STOP:
            break
        if fid == 1:
            d["type"] = p.readI32()
        elif fid == 2:
            d["type_length"] = p.readI32()
        elif fid == 3:
            d["repetition_type"] = p.readI32()
        elif fid == 4:
            d["name"] = p.readString()
        elif fid == 5:
            d["num_children"] = p.readI32()
        else:
            p.skip(t)
        p.readFieldEnd()
    p.readStructEnd()
    return d



def _statistics(p: TCompactProtocol) -> dict[str, Any]:
    d = {}
    p.readStructBegin()
    while True:
        _, t, fid = p.readFieldBegin()
        if t == TType.STOP:
            break
        if t == TType.STRING:
            d[str(fid)] = p.readBinary()
        elif t == TType.I64:
            d[str(fid)] = p.readI64()
        elif t == TType.BOOL:
            d[str(fid)] = p.readBool()
        else:
            p.skip(t)
        p.readFieldEnd()
    p.readStructEnd()
    return d

def _column_meta(p: TCompactProtocol) -> dict[str, Any]:
    d = {}
    p.readStructBegin()
    while True:
        _, t, fid = p.readFieldBegin()
        if t == TType.STOP:
            break
        if fid == 1:
            d["type"] = p.readI32()
        elif fid == 3:
            d["path"] = _list(p)
        elif fid == 4:
            d["codec"] = p.readI32()
        elif fid == 5:
            d["num_values"] = p.readI64()
        elif fid == 7:
            d["total_compressed_size"] = p.readI64()
        elif fid == 9:
            d["data_page_offset"] = p.readI64()
        elif fid == 11:
            d["dictionary_page_offset"] = p.readI64()
        elif fid == 12:
            d["statistics"] = _statistics(p)
        else:
            p.skip(t)
        p.readFieldEnd()
    p.readStructEnd()
    return d


def _column_chunk(p: TCompactProtocol) -> dict[str, Any]:
    d = {}
    p.readStructBegin()
    while True:
        _, t, fid = p.readFieldBegin()
        if t == TType.STOP:
            break
        if fid == 3:
            d["meta"] = _column_meta(p)
        elif fid == 2:
            d["file_offset"] = p.readI64()
        else:
            p.skip(t)
        p.readFieldEnd()
    p.readStructEnd()
    return d


def _row_group(p: TCompactProtocol) -> dict[str, Any]:
    d = {}
    p.readStructBegin()
    while True:
        _, t, fid = p.readFieldBegin()
        if t == TType.STOP:
            break
        if fid == 1:
            d["columns"] = _list(p, _column_chunk)
        elif fid == 3:
            d["num_rows"] = p.readI64()
        else:
            p.skip(t)
        p.readFieldEnd()
    p.readStructEnd()
    return d


def _file_meta(data: bytes) -> dict[str, Any]:
    p = TCompactProtocol(TMemoryBuffer(data))
    d = {}
    p.readStructBegin()
    while True:
        _, t, fid = p.readFieldBegin()
        if t == TType.STOP:
            break
        if fid == 2:
            d["schema"] = _list(p, _schema)
        elif fid == 3:
            d["num_rows"] = p.readI64()
        elif fid == 4:
            d["row_groups"] = _list(p, _row_group)
        else:
            p.skip(t)
        p.readFieldEnd()
    p.readStructEnd()
    return d


def _data_header(p: TCompactProtocol) -> dict[str, Any]:
    d = {}
    p.readStructBegin()
    while True:
        _, t, fid = p.readFieldBegin()
        if t == TType.STOP:
            break
        if fid == 1:
            d["num_values"] = p.readI32()
        elif fid == 2:
            d["encoding"] = p.readI32()
        elif fid == 3:
            d["definition_encoding"] = p.readI32()
        elif fid == 4:
            d["repetition_encoding"] = p.readI32()
        elif fid == 5:
            d["statistics"] = _statistics(p)
        else:
            p.skip(t)
        p.readFieldEnd()
    p.readStructEnd()
    return d


def _dict_header(p: TCompactProtocol) -> dict[str, Any]:
    d = {}
    p.readStructBegin()
    while True:
        _, t, fid = p.readFieldBegin()
        if t == TType.STOP:
            break
        if fid == 1:
            d["num_values"] = p.readI32()
        elif fid == 2:
            d["encoding"] = p.readI32()
        else:
            p.skip(t)
        p.readFieldEnd()
    p.readStructEnd()
    return d


def _page_header(data: bytes) -> tuple[dict[str, Any], int]:
    tr = TMemoryBuffer(data)
    p = TCompactProtocol(tr)
    d = {}
    p.readStructBegin()
    while True:
        _, t, fid = p.readFieldBegin()
        if t == TType.STOP:
            break
        if fid == 1:
            d["type"] = p.readI32()
        elif fid == 2:
            d["uncompressed_size"] = p.readI32()
        elif fid == 3:
            d["compressed_size"] = p.readI32()
        elif fid == 5:
            d["data"] = _data_header(p)
        elif fid == 7:
            d["dictionary"] = _dict_header(p)
        else:
            p.skip(t)
        p.readFieldEnd()
    p.readStructEnd()
    return d, tr.cstringio_buf.tell()


def _uvarint(data: bytes, pos: int = 0) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        b = data[pos]
        pos += 1
        value |= (b & 0x7F) << shift
        if not (b & 0x80):
            return value, pos
        shift += 7


def _snappy(data: bytes) -> bytes:
    expected, pos = _uvarint(data)
    out = bytearray()
    while pos < len(data):
        tag = data[pos]
        pos += 1
        kind = tag & 3
        if kind == 0:
            code = tag >> 2
            if code < 60:
                length = code + 1
            else:
                n = code - 59
                length = int.from_bytes(data[pos:pos+n], "little") + 1
                pos += n
            out.extend(data[pos:pos+length])
            pos += length
            continue
        if kind == 1:
            length = ((tag >> 2) & 7) + 4
            offset = ((tag & 0xE0) << 3) | data[pos]
            pos += 1
        elif kind == 2:
            length = (tag >> 2) + 1
            offset = int.from_bytes(data[pos:pos+2], "little")
            pos += 2
        else:
            length = (tag >> 2) + 1
            offset = int.from_bytes(data[pos:pos+4], "little")
            pos += 4
        for _ in range(length):
            out.append(out[-offset])
    if len(out) != expected:
        raise ValueError(f"Snappy size mismatch {len(out)} != {expected}")
    return bytes(out)


def _decompress(data: bytes, codec: int, expected: int) -> bytes:
    out = data if codec == UNCOMPRESSED else _snappy(data)
    if len(out) != expected:
        raise ValueError(f"Page size mismatch {len(out)} != {expected}")
    return out


def _hybrid(data: bytes, bit_width: int, count: int) -> list[int]:
    if bit_width == 0:
        return [0] * count
    out = []
    pos = 0
    width = (bit_width + 7) // 8
    mask = (1 << bit_width) - 1
    while len(out) < count:
        header, pos = _uvarint(data, pos)
        if header & 1 == 0:
            n = header >> 1
            value = int.from_bytes(data[pos:pos+width], "little") & mask
            pos += width
            out.extend([value] * min(n, count - len(out)))
        else:
            groups = header >> 1
            n_values = groups * 8
            n_bytes = groups * bit_width
            packed = int.from_bytes(data[pos:pos+n_bytes], "little")
            pos += n_bytes
            take = min(n_values, count - len(out))
            out.extend((packed >> (i * bit_width)) & mask for i in range(take))
    return out


def _plain(data: bytes, ptype: int, count: int) -> list[Any]:
    if ptype == BOOLEAN:
        return [bool((data[i // 8] >> (i % 8)) & 1) for i in range(count)]
    if ptype == BYTE_ARRAY:
        values = []
        pos = 0
        for _ in range(count):
            length = int.from_bytes(data[pos:pos + 4], "little")
            pos += 4
            raw = data[pos:pos + length]
            pos += length
            try:
                values.append(raw.decode("utf-8"))
            except UnicodeDecodeError:
                values.append(raw)
        return values
    fmt, width = {
        INT32: ("<i", 4),
        INT64: ("<q", 8),
        FLOAT: ("<f", 4),
        DOUBLE: ("<d", 8),
    }[ptype]
    return [v[0] for v in struct.iter_unpack(fmt, data[:count * width])]


def _decode_data(payload: bytes, header: dict[str, Any], ptype: int,
                 dictionary: list[Any] | None, optional: bool) -> list[Any]:
    n = header["num_values"]
    pos = 0
    if optional:
        size = int.from_bytes(payload[:4], "little")
        defs = _hybrid(payload[4:4+size], 1, n)
        pos = 4 + size
    else:
        defs = [0] * n
    present = sum(x == (1 if optional else 0) for x in defs)
    encoded = payload[pos:]
    if header["encoding"] == RLE_DICTIONARY:
        bw = encoded[0]
        idx = _hybrid(encoded[1:], bw, present)
        vals = [dictionary[i] for i in idx]
    elif header["encoding"] == PLAIN:
        vals = _plain(encoded, ptype, present)
    else:
        raise NotImplementedError(header["encoding"])
    it = iter(vals)
    max_def = 1 if optional else 0
    return [next(it) if d == max_def else None for d in defs]


def _read_column(blob: bytes, meta: dict[str, Any], schema: dict[str, Any]) -> list[Any]:
    start = meta.get("dictionary_page_offset", meta["data_page_offset"])
    end = start + meta["total_compressed_size"]
    pos = start
    dictionary = None
    out = []
    while pos < end and len(out) < meta["num_values"]:
        ph, hsize = _page_header(blob[pos:end])
        pos += hsize
        comp = blob[pos:pos+ph["compressed_size"]]
        pos += ph["compressed_size"]
        payload = _decompress(comp, meta["codec"], ph["uncompressed_size"])
        if ph["type"] == DICTIONARY_PAGE:
            dictionary = _plain(payload, meta["type"], ph["dictionary"]["num_values"])
        elif ph["type"] == DATA_PAGE:
            out.extend(_decode_data(
                payload, ph["data"], meta["type"], dictionary,
                schema.get("repetition_type") == 1,
            ))
        else:
            raise NotImplementedError(ph["type"])
    return out


def read_parquet(path: str | Path) -> pd.DataFrame:
    blob = Path(path).read_bytes()
    footer_size = int.from_bytes(blob[-8:-4], "little")
    meta = _file_meta(blob[-8-footer_size:-8])
    schemas = {x["name"]: x for x in meta["schema"][1:]}
    columns = {name: [] for name in schemas}
    for rg in meta["row_groups"]:
        for chunk in rg["columns"]:
            cm = chunk["meta"]
            name = cm["path"][0]
            columns[name].extend(_read_column(blob, cm, schemas[name]))
    df = pd.DataFrame(columns)
    for column in ("DateKey", "TargetDateKey", "origin"):
        if column in df and pd.api.types.is_numeric_dtype(df[column]):
            numeric = pd.to_numeric(df[column], errors="coerce")
            magnitude = float(numeric.abs().median()) if numeric.notna().any() else 0.0
            if magnitude >= 1e17:
                unit = "ns"
            elif magnitude >= 1e14:
                unit = "us"
            elif magnitude >= 1e11:
                unit = "ms"
            else:
                unit = "s"
            df[column] = pd.to_datetime(numeric, unit=unit)
    for c in ["ProductId", "CampaignSubTypeWeb", "CampaignSubTypeApp"]:
        if c in df:
            df[c] = df[c].astype("int64")
    for c in ["QuantityApp", "QuantityWeb"]:
        if c in df:
            df[c] = df[c].astype("int32")
    for c in ["ProductAvailable", "IsSaleOrPromo"]:
        if c in df:
            df[c] = df[c].astype(bool)
    for c in ["DiscountValueWebRelative", "DiscountValueAppRelative", "PriceLocalVat"]:
        if c in df:
            df[c] = df[c].astype("float32")
    if len(df) != meta["num_rows"]:
        raise ValueError(f"Decoded {len(df)}, expected {meta['num_rows']}")
    return df
