"""Argparse CLI configuration for HueCLI."""

import argparse
from config import COLOR_SCHEMES


def parse_args():
    """Parse CLI arguments and return argparse namespace."""
    parser = argparse.ArgumentParser(
        description='HueCLI - Generate multi-color 3D print STLs from images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Example: python3 huecli.py image.png --colors 5 --size 120x160'
    )

    parser.add_argument('image', help='Input image path (PNG/JPG/WEBP)')
    parser.add_argument('-f', '--filaments', type=str, default=None,
                        help='Filament library CSV path (default: filaments.csv)')
    parser.add_argument('-c', '--colors', type=int, default=None,
                        help='Number of colors/filaments (default: 4)')
    parser.add_argument('-n', '--nozzle', type=float, default=None,
                        help='Nozzle diameter in mm (default: 0.2)')
    parser.add_argument('-l', '--layer-height', type=float, default=None,
                        help='Layer height in mm (default: 0.08)')
    parser.add_argument('-m', '--model-height', type=float, default=None,
                        help='Total model height in mm (default: 2.0)')
    parser.add_argument('-s', '--size', type=str, default=None,
                        help='Print size as WIDTHxHEIGHT in mm, e.g. 100x140 (default: 100x140)')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output filename (default: <input_stem>.stl)')
    parser.add_argument('-d', '--min-delta-e', type=float, default=None,
                        help='Min color difference delta-E between filaments (default: 5.0)')
    parser.add_argument('--mode', type=str, default=None,
                        choices=['standard', 'flat', 'flat-cap',
                                 'exploded', 'exploded-multi', 'exploded-cmyk'],
                        help='Generation mode (default: standard)')
    parser.add_argument('--sandwich-layers', type=int, default=None,
                        help='Color layers per sandwich in exploded modes (default: 1, CMYK default: 3)')
    parser.add_argument('--fill', type=str, default=None, metavar='BOOL',
                        help='Fill sandwiches with transparent (inverse middle + top layer) [yes/no] (default: no)')
    parser.add_argument('--base-layers', type=int, default=None,
                        help='Transparent base layers per sandwich in exploded modes (default: 3)')
    parser.add_argument('--max-color-sandwiches', type=int, default=None,
                        help='Max sandwiches per colour in exploded modes (default: 3, multi: 5, CMYK: 1)')
    parser.add_argument('--scheme', type=str, default=None,
                        choices=list(COLOR_SCHEMES.keys()),
                        help='Remap image to a colour scheme palette before processing')
    parser.add_argument('--flip', type=str, default=None,
                        choices=['horizontal', 'vertical', 'both'],
                        help='Flip the image before processing (horizontal, vertical, or both)')
    parser.add_argument('--dither', type=str, default=None,
                        choices=['none', 'floyd-steinberg', 'ordered'],
                        help='Dithering for flat/exploded modes (default: none)')
    parser.add_argument('--enhance-detail', action='store_true', default=False,
                        help='Extend darkest filament through edges for sharper detail (standard mode only)')

    return parser.parse_args()
