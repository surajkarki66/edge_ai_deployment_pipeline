#!/usr/bin/env python3
# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  *
#  * Standalone TFLite PTQ extracted from stm32ai-modelzoo-services.
#  *
#  * This software component is provided to you as part of a software package and
#  * applicable license terms are in the Package_license file. If you received this
#  * software component outside of a package or without applicable license terms,
#  * the terms of the SLA0044 license shall apply and are fully reproduced below:
#  *
#  * SLA0044 Rev5/February 2018
#  *
#  * Software license agreement
#  *
#  * ULTIMATE LIBERTY SOFTWARE LICENSE AGREEMENT
#  *
#  * BY INSTALLING, COPYING, DOWNLOADING, ACCESSING OR OTHERWISE USING THIS SOFTWARE
#  * OR ANY PART THEREOF (AND THE RELATED DOCUMENTATION) FROM STMICROELECTRONICS
#  * INTERNATIONAL N.V, SWISS BRANCH AND/OR ITS AFFILIATED COMPANIES
#  * (STMICROELECTRONICS), THE RECIPIENT, ON BEHALF OF HIMSELF OR HERSELF, OR ON
#  * BEHALF OF ANY ENTITY BY WHICH SUCH RECIPIENT IS EMPLOYED AND/OR ENGAGED AGREES
#  * TO BE BOUND BY THIS SOFTWARE LICENSE AGREEMENT.
#  *
#  * Under STMicroelectronics' intellectual property rights, the redistribution,
#  * reproduction and use in source and binary forms of the software or any part
#  * thereof, with or without modification, are permitted provided that the following
#  * conditions are met:
#  *
#  * 1. Redistribution of source code (modified or not) must retain any copyright
#  * notice, this list of conditions and the disclaimer set forth below as items 10
#  * and 11.
#  *
#  * 2. Redistributions in binary form, except as embedded into microcontroller or
#  * microprocessor device manufactured by or for STMicroelectronics or a software
#  * update for such device, must reproduce any copyright notice provided with the
#  * binary code, this list of conditions, and the disclaimer set forth below as
#  * items 10 and 11, in documentation and/or other materials provided with the
#  * distribution.
#  *
#  * 3. Neither the name of STMicroelectronics nor the names of other contributors to
#  * this software may be used to endorse or promote products derived from this
#  * software or part thereof without specific written permission.
#  *
#  * 4. This software or any part thereof, including modifications and/or derivative
#  * works of this software, must be used and execute solely and exclusively on or in
#  * combination with a microcontroller or microprocessor device manufactured by or
#  * for STMicroelectronics.
#  *
#  * 5. No use, reproduction or redistribution of this software partially or totally
#  * may be done in any manner that would subject this software to any Open Source
#  * Terms. "Open Source Terms" shall mean any open source license which requires as
#  * part of distribution of software that the source code of such software is
#  * distributed therewith or otherwise made available, or open source license that
#  * substantially complies with the Open Source definition specified at
#  * www.opensource.org and any other comparable open source license such as for
#  * example GNU General Public License (GPL), Eclipse Public License (EPL), Apache
#  * Software License, BSD license or MIT license.
#  *
#  * 6. STMicroelectronics has no obligation to provide any maintenance, support or
#  * updates for the software.
#  *
#  * 7. The software is and will remain the exclusive property of STMicroelectronics
#  * and its licensors. The recipient will not take any action that jeopardizes
#  * STMicroelectronics and its licensors' proprietary rights or acquire any rights
#  * in the software, except the limited rights specified hereunder.
#  *
#  * 8. The recipient shall comply with all applicable laws and regulations affecting
#  * the use of the software or any part thereof including any applicable export
#  * control law or regulation.
#  *
#  * 9. Redistribution and use of this software or any part thereof other than as
#  * permitted under this license is void and will automatically terminate your
#  * rights under this license.
#  *
#  * 10. THIS SOFTWARE IS PROVIDED BY STMICROELECTRONICS AND CONTRIBUTORS "AS IS" AND
#  * ANY EXPRESS, IMPLIED OR STATUTORY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  * IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
#  * NON-INFRINGEMENT OF THIRD PARTY INTELLECTUAL PROPERTY RIGHTS, WHICH ARE
#  * DISCLAIMED TO THE FULLEST EXTENT PERMITTED BY LAW. IN NO EVENT SHALL
#  * STMICROELECTRONICS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
#  * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#  * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
#  * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
#  * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
#  * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
#  * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#  *
#  * 11. EXCEPT AS EXPRESSLY PERMITTED HEREUNDER, NO LICENSE OR OTHER RIGHTS, WHETHER
#  * EXPRESS OR IMPLIED, ARE GRANTED UNDER ANY PATENT OR OTHER INTELLECTUAL PROPERTY
#  * RIGHTS OF STMICROELECTRONICS OR ANY THIRD PARTY.
#  *--------------------------------------------------------------------------------------------*/

# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys

from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

PACKAGE_ROOT = Path(__file__).resolve().parent

# Board name -> stmaic config file in --c-project
BOARD_CONF: dict[str, str] = {
    "STM32N6570-DK": "stmaic_STM32N6570-DK.conf",
}
DEFAULT_BOARD = "STM32N6570-DK"


def stmaic_conf_for_board(board: str) -> str:
    conf = BOARD_CONF.get(board)
    if not conf:
        supported = ", ".join(BOARD_CONF)
        raise ValueError(f"Unsupported board '{board}'. Supported: {supported}")
    return conf

ASPECT_RATIO_N6 = {
    "fit": "ASPECT_RATIO_FIT",
    "crop": "ASPECT_RATIO_CROP",
    "full_screen": "ASPECT_RATIO_FULLSCREEN",
}
COLOR_MODE_N6 = {"rgb": "COLOR_RGB", "bgr": "COLOR_BGR"}

# Typical STEdgeAI Core install locations
_STEDGEAI_SEARCH_ROOTS = (
    Path("/opt/ST"),
    Path(os.environ.get("STEDGEAI_ROOT", "")),
)


def resolve_stedgeai_path(user_path: str) -> Optional[Path]:
    """Find stedgeai executable: explicit path, STM32_AI_EXE, or under /opt/ST."""
    candidates: List[Path] = []
    if user_path:
        candidates.append(Path(user_path))
    env = os.environ.get("STM32_AI_EXE", "")
    if env:
        candidates.append(Path(env))

    for root in _STEDGEAI_SEARCH_ROOTS:
        if not root or not root.is_dir():
            continue
        for pattern in (
            "STEdgeAI/*/Utilities/linux/stedgeai",
            "STEdgeAI/*/Utilities/windows/stedgeai.exe",
            "**/stedgeai",
        ):
            candidates.extend(sorted(root.glob(pattern)))

    seen = set()
    for p in candidates:
        p = p.resolve()
        if p in seen or not p.is_file():
            continue
        seen.add(p)
        if os.access(p, os.X_OK):
            return p
    return None


def ensure_package_path() -> None:
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))


def parse_class_names(value: str) -> List[str]:
    """Labels for app_config.h only."""
    names = [s.strip() for s in value.split(",") if s.strip()]
    if not names:
        raise ValueError("--class-names must contain at least one non-empty label")
    return names


def generate_n6_headers(
    output_dir: Path,
    model_path: Path,
    class_names: List[str],
    aspect_ratio: str,
    color_mode: str,
) -> Path:
    """app_config.h for Getting Started."""
    header_dir = output_dir / "C_header"
    header_dir.mkdir(parents=True, exist_ok=True)

    classes = "{\\\n"
    for i, name in enumerate(class_names):
        if i == len(class_names) - 1:
            classes += f'   "{name}"' + "}\\"
        else:
            classes += f'   "{name}"' + " ,"
            if i % 5 == 0 and i != 0:
                classes += "\\\n"

    aspect = ASPECT_RATIO_N6.get(aspect_ratio, "ASPECT_RATIO_FIT")
    color = COLOR_MODE_N6.get(color_mode, "COLOR_RGB")

    app_config = header_dir / "app_config.h"
    with open(app_config, "wt", encoding="utf-8") as f:
        f.write("/**\n")
        f.write("******************************************************************************\n")
        f.write("* @file    app_config.h\n")
        f.write("* @author  GPM Application Team\n")
        f.write("******************************************************************************\n")
        f.write("*/\n\n")
        f.write("#ifndef APP_CONFIG\n#define APP_CONFIG\n\n")
        f.write("#define USE_DCACHE\n\n")
        f.write('#include "arm_math.h"\n\n')
        f.write("#define CAMERA_FLIP CMW_MIRRORFLIP_NONE\n\n")
        f.write("#define ASPECT_RATIO_CROP (1)\n")
        f.write("#define ASPECT_RATIO_FIT (2)\n")
        f.write("#define ASPECT_RATIO_FULLSCREEN (3)\n")
        f.write(f"#define ASPECT_RATIO_MODE    {aspect}\n\n")
        f.write("#define COLOR_BGR (0)\n#define COLOR_RGB (1)\n")
        f.write(f"#define COLOR_MODE {color}\n\n")
        f.write(f"#define NB_CLASSES   ({len(class_names)})\n")
        f.write(f"#define CLASSES_TABLE const char* classes_table[NB_CLASSES] = {classes}\n\n")
        f.write(f'#define WELCOME_MSG_1     "{model_path.name}"\n')
        f.write('#define WELCOME_MSG_2       "Model Running in STM32 MCU internal memory"\n')
        f.write("\n#endif      /* APP_CONFIG */\n")

    logger.info("Wrote %s", app_config)
    return header_dir


def check_c_project(c_project: Path, board: str) -> str:
    conf_name = stmaic_conf_for_board(board)
    conf_path = c_project / conf_name
    if not c_project.is_dir() or not any(c_project.iterdir()):
        raise FileNotFoundError(
            f"C project empty or missing: {c_project}\n"
            "Clone the STM32N6 Getting Started Image Classification project and pass --c-project:\n"
            "  https://github.com/STMicroelectronics/STM32N6-GettingStarted-ImageClassification"
        )
    if not conf_path.is_file():
        raise FileNotFoundError(f"Board config not found: {conf_path}")
    return conf_name


def compile_and_optional_build(
    model_path: Path,
    output_dir: Path,
    c_project_path: Path,
    board: str,
    stedgeai_path: str,
    cube_ide_path: Optional[str],
    class_names: List[str],
    aspect_ratio: str,
    color_mode: str,
    optimization: str,
    do_build: bool,
    stlink_serial: Optional[str],
) -> Path:
    ensure_package_path()
    import edge_ai_deployment_pipeline.adapters.mcu.toolchains.stm32n6.image_classification.common.stm32ai_local as stmaic

    conf_name = check_c_project(c_project_path, board)

    os.environ["STM32_AI_EXE"] = stedgeai_path
    if cube_ide_path:
        os.environ["STM32_CUBE_IDE_EXE"] = cube_ide_path

    stm32ai_output = output_dir / "generated"
    export_subdir = "quantized_models"
    shutil.rmtree(stm32ai_output, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    generate_n6_headers(
        output_dir=output_dir,
        model_path=model_path,
        class_names=class_names,
        aspect_ratio=aspect_ratio,
        color_mode=color_mode,
    )

    session = stmaic.load(str(model_path), workspace_dir=str(output_dir))
    tools = stmaic.STMAiTools()
    session.set_tools(tools)

    board_conf_path = c_project_path / conf_name
    build_conf = None
    board_cfg = stmaic.STMAiBoardConfig(str(board_conf_path), build_conf)
    session.set_board(board_cfg)

    neural_art_path = (
        board_cfg.config.profile + "@" + board_cfg.config.neuralart_user_path
    )
    opt = stmaic.STMAiCompileOptions(
        st_neural_art=neural_art_path,
        input_data_type="uint8",
        output_data_type="float32",
        inputs_ch_position="chlast",
        outputs_ch_position="",
        name="network",
        no_outputs_allocation=False,
        no_inputs_allocation=False,
        optimization=optimization,
    )

    logger.info("Compiling model with STEdgeAI (STM32N6)...")
    logger.info("Model: %s", model_path)
    logger.info("Board config: %s", board_conf_path)
    logger.info("STEdgeAI optimization (-O): %s", optimization)
    stmaic.compile(session=session, options=opt, target=session._board_config)

    tflite_in_export = output_dir / export_subdir / model_path.name
    if model_path.resolve() != tflite_in_export.resolve():
        dest = output_dir / export_subdir
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_path, dest / model_path.name)

    generated_tflite = stm32ai_output
    if not generated_tflite.is_dir():
        raise RuntimeError(f"STEdgeAI did not create {generated_tflite}")

    logger.info("Generated C code: %s", generated_tflite)
    for name in ("stai_network.c", "stai_network.h", "network_atonbuf.xSPI2.raw"):
        p = output_dir / name
        src = generated_tflite / name
        if src.is_file():
            shutil.copy2(src, p)

    if not do_build:
        logger.info("Compile-only done (skip firmware build). Use --build to build/flash.")
        return generated_tflite

    if not cube_ide_path:
        raise ValueError("--build requires --cube-ide (or STM32_CUBE_IDE_EXE)")

    user_files = [
        str(output_dir / "C_header/app_config.h"),
    ]
    for extra in (
        "stai_network.c",
        "stai_network.h",
        "network_atonbuf.xSPI2.raw",
    ):
        p = output_dir / extra
        if p.is_file():
            user_files.append(str(p))

    logger.info("Building and flashing CubeIDE project...")
    stmaic.build(session, user_files=user_files, serial_number=stlink_serial)
    return generated_tflite


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Compile .tflite to STM32N6 C code",
    )
    parser.add_argument("--model", required=True, help="Quantized .tflite model")
    parser.add_argument(
        "--output-dir",
        default="./stm32n6_compiled",
        help="Workspace (generated/, C_header/, etc.)",
    )
    parser.add_argument(
        "--c-project",
        required=True,
        help="Path to STM32N6 Getting Started CubeIDE project (STM32N6-GettingStarted-ImageClassification)",
    )
    parser.add_argument(
        "--board",
        default=DEFAULT_BOARD,
        choices=list(BOARD_CONF.keys()),
        help=f"Target board (see BOARD_CONF in script; default: {DEFAULT_BOARD})",
    )
    parser.add_argument(
        "--class-names",
        required=True,
        help="Comma-separated labels for on-device UI in app_config.h (not used by STEdgeAI compile)",
    )
    parser.add_argument(
        "--aspect-ratio",
        default="crop",
        choices=list(ASPECT_RATIO_N6.keys()),
        help="Camera/display preprocessing in app (deployment_n6_config default: crop)",
    )
    parser.add_argument("--color-mode", default="rgb", choices=["rgb", "bgr"])
    parser.add_argument(
        "--stedgeai",
        default=os.environ.get("STM32_AI_EXE", ""),
        help="Path to stedgeai executable (or set STM32_AI_EXE)",
    )
    parser.add_argument(
        "--cube-ide",
        default=os.environ.get("STM32_CUBE_IDE_EXE", ""),
        help="Path to STM32CubeIDE (required for --build)",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Build and flash the Getting Started firmware (not just generate C)",
    )
    parser.add_argument("--stlink-serial", default=None, help="ST-LINK serial number")
    parser.add_argument(
        "--optimization",
        default="balanced",
        choices=["balanced", "time", "ram"],
        help="STEdgeAI -O flag (balanced, time, ram); NeuralART profile in board conf still applies",
    )
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    output_dir = Path(args.output_dir).resolve()
    c_project = Path(args.c_project).resolve()

    if not model_path.is_file() or model_path.suffix.lower() != ".tflite":
        logger.error("--model must be an existing .tflite file")
        return 1

    stedgeai_exe = resolve_stedgeai_path(args.stedgeai)
    if not stedgeai_exe:
        logger.error(
            "STEdgeAI CLI not found.\n"
            "  Install adds the binary under /opt/ST/STEdgeAI/<ver>/Utilities/linux/stedgeai\n"
            "  (it is usually NOT on your PATH).\n\n"
            "  Use either:\n"
            "    export STM32_AI_EXE=/opt/ST/STEdgeAI/4.0/Utilities/linux/stedgeai\n"
            "    python compile_model.py --stedgeai /opt/ST/STEdgeAI/4.0/Utilities/linux/stedgeai ..."
        )
        return 1
    logger.info("Using STEdgeAI: %s", stedgeai_exe)

    class_names = parse_class_names(args.class_names)

    try:
        compile_and_optional_build(
            model_path=model_path,
            output_dir=output_dir,
            c_project_path=c_project,
            board=args.board,
            stedgeai_path=str(stedgeai_exe),
            cube_ide_path=args.cube_ide or None,
            class_names=class_names,
            aspect_ratio=args.aspect_ratio,
            color_mode=args.color_mode,
            optimization=args.optimization,
            do_build=args.build,
            stlink_serial=args.stlink_serial,
        )
    except Exception as exc:
        logger.error("%s", exc)
        return 1

    logger.info("Done. Output workspace: %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
