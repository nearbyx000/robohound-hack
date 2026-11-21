"""Generate a collision scene retaining the published geom and body indices."""

from pathlib import Path
import xml.etree.ElementTree as ET


def prepare(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    robot = ET.parse(source / "upstream-go1.xml")
    # Visualization meshes are excluded from all 655 judge pairs. Replace them
    # with disabled placeholders to retain the original geom IDs without STLs.
    robot.find("compiler").attrib.pop("meshdir")
    robot.getroot().remove(robot.find("asset"))
    visual_default = robot.find(".//default[@class='visual']/geom")
    visual_default.attrib.pop("material")
    visual_default.set("type", "sphere")
    visual_default.set("size", "0.001")
    for geom in robot.findall(".//worldbody//geom[@mesh]"):
        if geom.get("class") != "visual":
            raise ValueError("Cannot replace a collision mesh with a placeholder")
        geom.attrib.pop("mesh")
    robot.write(destination / "go1.xml", encoding="utf-8")

    scene = ET.parse(source / "upstream-scene.xml")
    scene.getroot().remove(scene.find("visual"))
    scene.getroot().remove(scene.find("asset"))
    scene.find(".//geom[@name='floor']").attrib.pop("material")
    scene.write(destination / "scene.xml", encoding="utf-8")


if __name__ == "__main__":
    import sys

    prepare(Path(__file__).resolve().parent, Path(sys.argv[1]))
