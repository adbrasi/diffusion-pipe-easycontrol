from tools.nextscene_captions import action_caption


def test_removes_shared_appearance_but_preserves_action():
    caption = 'close-up, profile view, a person turning quickly to the left wearing a dark blue hat and a tunic.'
    assert action_caption(caption) == 'close-up, profile view. The character is turning quickly to the left.'


def test_does_not_turn_credits_into_character_action():
    assert action_caption('A male character with glowing eyes, floating white text credits visible in the foreground.') is None


def test_leaves_no_incomplete_article_after_cutting_appearance():
    caption = 'Close-up, a girl with long blue hair and a gray jacket looking forward with a surprised hair silhouette, wearing a silver necklace.'
    short = action_caption(caption)
    assert short.endswith('with a surprised expression.')


def test_no_action_uses_only_original_caption():
    assert action_caption('An empty stone castle seen from a high angle, surrounded by a forest.') is None
