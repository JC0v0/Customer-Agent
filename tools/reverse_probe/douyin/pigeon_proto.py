# -*- coding: utf-8 -*-
"""极简 protobuf wire 编解码器（pigeon 帧专用）。"""
import io


def enc_varint(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            break
    return bytes(out)


def enc_tag(field, wt):
    return enc_varint((field << 3) | wt)


def enc_varint_field(field, val):
    return enc_tag(field, 0) + enc_varint(val)


def enc_bytes_field(field, data):
    return enc_tag(field, 2) + enc_varint(len(data)) + bytes(data)


def dec_varint(data, pos):
    r = 0; s = 0
    while True:
        b = data[pos]; pos += 1
        r |= (b & 0x7F) << s
        if not (b & 0x80):
            break
        s += 7
    return r, pos


def parse_fields(data):
    """返回 [(field, wiretype, value, start, end)]，value 依 wiretype 而定。"""
    out = []
    pos = 0
    while pos < len(data):
        tag, pos = dec_varint(data, pos)
        f = tag >> 3; wt = tag & 7
        if wt == 0:
            v, pos = dec_varint(data, pos)
            out.append((f, 0, v, None))
        elif wt == 2:
            ln, pos = dec_varint(data, pos)
            chunk = data[pos:pos + ln]
            out.append((f, 2, chunk, pos))
            pos += ln
        elif wt == 5:
            out.append((f, 5, data[pos:pos + 4], pos)); pos += 4
        elif wt == 1:
            out.append((f, 1, data[pos:pos + 8], pos)); pos += 8
        elif wt == 6:
            # fixed64（8 字节小端）
            out.append((f, 6, int.from_bytes(data[pos:pos + 8], "little"), pos)); pos += 8
        elif wt == 3:
            # group start：跳过到对应 end(wt=4)
            start = pos
            depth = 1
            while depth > 0 and pos < len(data):
                tag, npos = dec_varint(data, pos)
                w = tag & 7
                pos = npos
                if w == 0:
                    _, pos = dec_varint(data, pos)
                elif w == 1:
                    pos += 8
                elif w == 2:
                    ln, pos = dec_varint(data, pos)
                    pos += ln
                elif w == 5:
                    pos += 4
                elif w == 6:
                    pos += 8
                elif w == 3:
                    depth += 1
                elif w == 4:
                    depth -= 1
            out.append((f, 3, b"", start))
        else:
            raise ValueError("unsupported wiretype %d" % wt)
    return out


def rebuild(fields):
    """将 parse_fields 的结果按原序重新编码（value 不变）。"""
    out = bytearray()
    for f, wt, v, _ in fields:
        if wt == 0:
            out += enc_varint_field(f, v)
        elif wt == 2:
            out += enc_bytes_field(f, v)
        elif wt == 5:
            out += enc_tag(f, 5) + bytes(v)
        elif wt == 1:
            out += enc_tag(f, 1) + bytes(v)
    return bytes(out)


def replace_bytes_field(data, target_field, new_value):
    """替换某字段（wt=2）的 value 为 new_value，重算长度。"""
    fields = parse_fields(data)
    done = False
    for i, (f, wt, v, _) in enumerate(fields):
        if f == target_field and wt == 2:
            fields[i] = (f, wt, bytes(new_value), None)
            done = True
            break
    if not done:
        raise KeyError("field %d not found" % target_field)
    return rebuild(fields)


def get_bytes_field(data, target_field):
    for f, wt, v, _ in parse_fields(data):
        if f == target_field and wt == 2:
            return v
    return None
