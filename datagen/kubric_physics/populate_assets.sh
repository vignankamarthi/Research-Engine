#!/bin/bash
# Populate the GSO + HDRI asset cache for the bpy generator by pulling the public
# Kubric asset tarballs from GCS over HTTPS (no gsutil / kubric needed). Idempotent:
# skips an asset already unpacked. Layout matches what kubric's AssetSource produced,
# so physics_gen_bpy.py finds visual_geometry.obj / object.urdf / environment_4k.hdr.
#
# Run on a node with internet (cpu compute node is most reliable):
#   srun --partition=cpu --cpus-per-task=4 --mem=8G --time=00:30:00 \
#     bash populate_assets.sh /work/neu/p2026_0016_neu/kubric/asset_cache
set -uo pipefail
CACHE=${1:-/work/neu/p2026_0016_neu/kubric/asset_cache}
GSO_BASE=https://storage.googleapis.com/kubric-public/assets/GSO
HDRI_BASE=https://storage.googleapis.com/kubric-public/assets/HDRI_haven

GSO_OBJECTS=(
  Nickelodeon_Teenage_Mutant_Ninja_Turtles_Leonardo
  Digital_Camo_Double_Decker_Lunch_Bag Mad_Gab_Refresh_Card_Game
  Creatine_Monohydrate ACE_Coffee_Mug_Kristen_16_oz_cup
  BIA_Cordon_Bleu_White_Porcelain_Utensil_Holder_900028
  UGG_Cambridge_Womens_Black_7 Sootheze_Cold_Therapy_Elephant
  30_CONSTRUCTION_SET
  Playmates_Industrial_CoSplinter_Teenage_Mutant_Ninja_Turtle_Action_Figure
  Olive_Kids_Birdie_Munch_n_Lunch
  Don_Franciscos_Gourmet_Coffee_Medium_Decaf_100_Colombian_12_oz_340_g
  Aroma_Stainless_Steel_Milk_Frother_2_Cup BlackBlack_Nintendo_3DSXL
  3D_Dollhouse_Swing Olive_Kids_Butterfly_Garden_Munch_n_Lunch_Bag
  IsoRich_Soy Central_Garden_Flower_Pot_Goo_425 BALANCING_CACTUS
  Organic_Whey_Protein_Unflavored Circo_Fish_Toothbrush_Holder_14995988
  Cole_Hardware_Antislip_Surfacing_Material_White
)
HDRI_ENVS=(
  abandoned_games_room_01 abandoned_workshop aerodynamics_workshop
  aft_lounge aircraft_workshop_01 anniversary_lounge art_studio
  artist_workshop ballroom cayley_interior christmas_photo_studio_01
  abandoned_hall_01
)

fetch() {  # url  dest_dir  sentinel_file
  local url=$1 dir=$2 sentinel=$3
  if [ -f "$dir/$sentinel" ]; then echo "skip  $(basename "$dir")"; return 0; fi
  mkdir -p "$dir"
  local tmp; tmp=$(mktemp)
  if curl -fsSL --max-time 300 --retry 4 -o "$tmp" "$url"; then
    tar xf "$tmp" -C "$dir" && echo "OK    $(basename "$dir")" || echo "FAIL-extract $(basename "$dir")"
  else
    echo "FAIL-download $(basename "$dir") ($url)"
  fi
  rm -f "$tmp"
}

echo "populating GSO -> $CACHE/GSO"
for a in "${GSO_OBJECTS[@]}"; do
  fetch "$GSO_BASE/$a.tar.gz" "$CACHE/GSO/$a" visual_geometry.obj
done
echo "populating HDRI -> $CACHE/HDRI"
for h in "${HDRI_ENVS[@]}"; do
  fetch "$HDRI_BASE/$h.tar.gz" "$CACHE/HDRI/$h" environment_4k.hdr
done

echo "=== coverage ==="
g=0; for a in "${GSO_OBJECTS[@]}"; do [ -f "$CACHE/GSO/$a/visual_geometry.obj" ] && g=$((g+1)); done
hn=0; for h in "${HDRI_ENVS[@]}"; do [ -f "$CACHE/HDRI/$h/environment_4k.hdr" ] && hn=$((hn+1)); done
echo "GSO unpacked: $g/${#GSO_OBJECTS[@]}   HDRI unpacked: $hn/${#HDRI_ENVS[@]}"
