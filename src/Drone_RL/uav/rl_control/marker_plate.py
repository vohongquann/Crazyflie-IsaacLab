"""Landing pad: a flat square with the ArUco marker as its texture (visual only, no collision).

The material shows the texture through the emissive colour as well, so the marker looks the same whatever the light,
as a printed marker under a steady light does for the real camera.
"""
from __future__ import annotations

from dataclasses import MISSING

from isaaclab.sim.spawners import SpawnerCfg
from isaaclab.sim.utils import clone, create_prim, get_current_stage
from isaaclab.utils import configclass

from Drone_RL.uav.mdp.aruco import MARKER_TEXTURE


@clone
def spawn_aruco_plate(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Spawn a ``cfg.size`` [m] square in the xy plane, facing +z, with the marker texture."""
    from pxr import Sdf, UsdGeom, UsdShade

    stage = get_current_stage()
    create_prim(prim_path, "Xform", translation=translation, orientation=orientation, stage=stage)

    half = cfg.size / 2.0
    mesh = UsdGeom.Mesh.Define(stage, f"{prim_path}/plate")
    mesh.CreatePointsAttr([(-half, -half, 0.0), (half, -half, 0.0), (half, half, 0.0), (-half, half, 0.0)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateNormalsAttr([(0.0, 0.0, 1.0)] * 4)
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    uv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    uv.Set([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])

    material = UsdShade.Material.Define(stage, f"{prim_path}/material")
    surface = UsdShade.Shader.Define(stage, f"{prim_path}/material/surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    texture = UsdShade.Shader.Define(stage, f"{prim_path}/material/texture")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(cfg.texture))
    texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
    texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
    reader = UsdShade.Shader.Define(stage, f"{prim_path}/material/uv_reader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        reader.ConnectableAPI(), reader.CreateOutput("result", Sdf.ValueTypeNames.Float2).GetBaseName())
    color = texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(), color.GetBaseName())
    surface.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(), color.GetBaseName())
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(mesh).Bind(material)
    return stage.GetPrimAtPath(prim_path)


@configclass
class ArucoPlateCfg(SpawnerCfg):
    func: callable = spawn_aruco_plate
    size: float = MISSING
    """Side of the square [m], white margin included."""
    texture: str = str(MARKER_TEXTURE)
