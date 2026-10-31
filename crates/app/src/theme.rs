//! Dark GUI theme: near-black background like a camera viewer, one amber accent that
//! matches the yellow-to-red overlay.

use eframe::egui::{
    self, Color32, CornerRadius, FontId, Margin, Stroke, TextStyle, Theme, Vec2, vec2,
};

/// Behind the camera image
pub const BACKGROUND: Color32 = Color32::from_rgb(0x0f, 0x11, 0x13);
/// Top bar
pub const PANEL: Color32 = Color32::from_rgb(0x18, 0x1b, 0x1f);
/// Widget fill
const SURFACE: Color32 = Color32::from_rgb(0x23, 0x27, 0x2d);
const SURFACE_HOVER: Color32 = Color32::from_rgb(0x2c, 0x31, 0x38);
const BORDER: Color32 = Color32::from_rgb(0x30, 0x35, 0x3c);
pub const TEXT: Color32 = Color32::from_rgb(0xe6, 0xe8, 0xeb);
pub const TEXT_WEAK: Color32 = Color32::from_rgb(0x9a, 0xa0, 0xa6);
pub const ACCENT: Color32 = Color32::from_rgb(0xf5, 0xa5, 0x24);
pub const OK: Color32 = Color32::from_rgb(0x3f, 0xb9, 0x50);
pub const ERROR: Color32 = Color32::from_rgb(0xf8, 0x51, 0x49);
/// Estimated source positions, contrasts with the amber overlay
pub const PEAK: Color32 = Color32::from_rgb(0x2f, 0x81, 0xf7);

/// Rounding of buttons, combo boxes and popups
const RADIUS: u8 = 6;

/// Apply the theme to all styles of `ctx` and force dark mode.
pub fn apply(ctx: &egui::Context) {
    ctx.set_theme(Theme::Dark);
    ctx.all_styles_mut(|style| {
        style.text_styles = [
            (TextStyle::Heading, FontId::proportional(20.0)),
            (TextStyle::Body, FontId::proportional(15.0)),
            (TextStyle::Button, FontId::proportional(15.0)),
            (TextStyle::Monospace, FontId::monospace(14.0)),
            (TextStyle::Small, FontId::proportional(12.0)),
        ]
        .into();

        let spacing = &mut style.spacing;
        spacing.item_spacing = vec2(10.0, 8.0);
        spacing.button_padding = vec2(12.0, 6.0);
        spacing.interact_size = Vec2::new(spacing.interact_size.x, 30.0);
        spacing.combo_width = 180.0;
        spacing.menu_margin = Margin::same(6);

        let visuals = &mut style.visuals;
        visuals.dark_mode = true;
        visuals.override_text_color = Some(TEXT);
        visuals.panel_fill = BACKGROUND;
        visuals.window_fill = PANEL;
        visuals.extreme_bg_color = BACKGROUND;
        visuals.faint_bg_color = SURFACE;
        visuals.window_stroke = Stroke::new(1.0, BORDER);
        visuals.window_corner_radius = CornerRadius::same(RADIUS + 2);
        visuals.menu_corner_radius = CornerRadius::same(RADIUS + 2);
        visuals.selection.bg_fill = ACCENT.gamma_multiply(0.25);
        visuals.selection.stroke = Stroke::new(1.0, ACCENT);

        let widgets = &mut visuals.widgets;
        widgets.noninteractive.bg_fill = PANEL;
        widgets.noninteractive.bg_stroke = Stroke::new(1.0, BORDER);
        widgets.noninteractive.fg_stroke = Stroke::new(1.0, TEXT_WEAK);
        for (state, fill, stroke) in [
            (&mut widgets.inactive, SURFACE, BORDER),
            (
                &mut widgets.hovered,
                SURFACE_HOVER,
                ACCENT.gamma_multiply(0.6),
            ),
            (&mut widgets.active, SURFACE_HOVER, ACCENT),
            (&mut widgets.open, SURFACE_HOVER, ACCENT),
        ] {
            state.bg_fill = fill;
            state.weak_bg_fill = fill;
            state.bg_stroke = Stroke::new(1.0, stroke);
            state.fg_stroke = Stroke::new(1.0, TEXT);
            state.corner_radius = CornerRadius::same(RADIUS);
            state.expansion = 0.0;
        }
    });
}
