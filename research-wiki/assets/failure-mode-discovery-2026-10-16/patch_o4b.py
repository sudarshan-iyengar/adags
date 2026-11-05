"""Apply the O4b colour patch to a COPY of a method's repo (train.py only). Usage: patch_o4b.py {4dgs|ex4dgs} REPO_COPY
Every anchor must occur exactly once; the script refuses otherwise and prints the unified diff it applied."""
import difflib, re, sys

kind, repo = sys.argv[1:3]
p = repo + "/train.py"
src = open(p).read()
s = src


def after_line(anchor, insert):
    """insert text after the (single) line containing anchor"""
    global s
    n = s.count(anchor)
    if n != 1:
        sys.exit("REFUSE: anchor found %d times: %r" % (n, anchor[:70]))
    j = s.index("\n", s.index(anchor)) + 1
    s = s[:j] + insert + s[j:]


def step_after_first_optimizer_step():
    """insert _CC.step() right after the first gaussians.optimizer.step() line, with that line's indentation"""
    global s
    m = re.search(r"^([ \t]*)gaussians\.optimizer\.step\(\)[^\n]*\n", s, re.M)
    if not m:
        sys.exit("REFUSE: no gaussians.optimizer.step()")
    s = s[:m.end()] + m.group(1) + "_CC.step()  # O4b\n" + s[m.end():]


first_import = re.search(r"^import [^\n]*$", s, re.M).group(0)
after_line(first_import, "from colourcorr import CamColour  # O4b oracle patch\n_CC = None  # created on first use (needs CUDA)\n")
if kind == "4dgs":
    ind = " " * 12
    after_line('image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"]',
               ind + "global _CC\n" + ind + "_CC = _CC or CamColour()\n"
               + ind + "image = _CC.apply(image, viewpoint_cam.colmap_id // 300)  # O4b: camera = image index // 300 frames\n")
    after_line("        loss = Ll1\n", "        loss = loss + _CC.reg_loss()  # O4b\n")
elif kind == "ex4dgs":
    ind = " " * 8
    after_line('render_pkg["radii"], render_pkg["depth"], render_pkg["opticalflow"], render_pkg["acc"], render_pkg["dominent_idxs"]',
               ind + "global _CC\n" + ind + "_CC = _CC or CamColour()\n"
               + ind + "image = _CC.apply(image, viewpoint_cam.colmap_id)  # O4b: colmap camera id\n")
    after_line("        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))\n",
               "        loss = loss + _CC.reg_loss()  # O4b\n")
else:
    sys.exit("kind must be 4dgs or ex4dgs")
step_after_first_optimizer_step()
open(p, "w").write(s)
sys.stdout.writelines(difflib.unified_diff(src.splitlines(True), s.splitlines(True), "train.py", "train.py (O4b)"))
