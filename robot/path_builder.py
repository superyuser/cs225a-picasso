from config import DRAW_STEP_M, INIT_POS, RETRACT_VEC, TRAVEL_STEP_M
from interpolation import make_position_segment


def append_segment(path, labels, p0, p1, step_m, label):
    """
    Append interpolated segment to global path.

    Avoids duplicating the first point when path is non-empty.
    """
    segment = make_position_segment(p0, p1, step_m=step_m)

    if len(path) > 0 and len(segment) > 0:
        segment = segment[1:]

    for p in segment:
        path.append(p)
        labels.append(label)


def build_drawing_path(_current_pos, strokes):
    """
    Build one flattened position-only path.

    Motion plan:
      start at INIT_POS
      for each stroke:
        travel to stroke_start + RETRACT_VEC
        move forward to stroke_start
        draw all stroke points
        retract from stroke_end to stroke_end + RETRACT_VEC
      return to INIT_POS
    """
    path = []
    labels = []

    p_cursor = INIT_POS.copy()

    # 2. Stroke execution.
    for stroke_idx, stroke in enumerate(strokes):
        pts = stroke["points"]

        stroke_start = pts[0]
        stroke_end = pts[-1]

        start_retracted = stroke_start + RETRACT_VEC
        end_retracted = stroke_end + RETRACT_VEC

        feature = stroke["feature"]
        order = stroke["order"]

        # Travel lifted/retracted to the start of this stroke.
        append_segment(
            path,
            labels,
            p_cursor,
            start_retracted,
            step_m=TRAVEL_STEP_M,
            label=f"stroke_{order}_{feature}_travel_to_retracted_start"
        )
        p_cursor = start_retracted.copy()

        # Move forward in +x into contact at stroke start.
        append_segment(
            path,
            labels,
            p_cursor,
            stroke_start,
            step_m=DRAW_STEP_M,
            label=f"stroke_{order}_{feature}_approach_contact"
        )
        p_cursor = stroke_start.copy()

        # Draw along the stroke.
        for j in range(1, len(pts)):
            append_segment(
                path,
                labels,
                p_cursor,
                pts[j],
                step_m=DRAW_STEP_M,
                label=f"stroke_{order}_{feature}_drawing"
            )
            p_cursor = pts[j].copy()

        # Retract in -x from stroke end.
        append_segment(
            path,
            labels,
            p_cursor,
            end_retracted,
            step_m=DRAW_STEP_M,
            label=f"stroke_{order}_{feature}_retract"
        )
        p_cursor = end_retracted.copy()

    # 3. Return to INIT from final retracted point.
    append_segment(
        path,
        labels,
        p_cursor,
        INIT_POS,
        step_m=TRAVEL_STEP_M,
        label="return_to_INIT"
    )

    return path, labels
