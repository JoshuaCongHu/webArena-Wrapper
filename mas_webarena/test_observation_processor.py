#!/usr/bin/env python3
"""Test the ObservationProcessor as specified in CLAUDE.md"""

from utils.observation_processor import ObservationProcessor


def test_accessibility_tree_processing():
    """Test processing of accessibility tree observations"""
    print("Testing accessibility tree processing...")
    
    processor = ObservationProcessor()
    
    # Sample accessibility tree observation (WebArena-style format)
    sample_obs = """[1] <button text="Search" clickable>
[2] <input type="text" text="" placeholder="Enter search term">
[3] <div text="Welcome to our website">
[4] <a href="/login" text="Login" clickable>
[5] <select>
[6] <option text="Option 1">
[7] <textarea placeholder="Comments">
[8] <button text="Submit Form" clickable>"""
    
    result = processor.process(sample_obs)
    
    print(f"Type: {result['type']}")
    print(f"Number of elements: {result['num_elements']}")
    print(f"Has form: {result['has_form']}")
    print(f"Has button: {result['has_button']}")
    print(f"Elements found: {len(result['elements'])}")
    
    # Verify basic structure
    assert result['type'] == 'accessibility_tree', "Should be accessibility_tree type"
    assert result['num_elements'] > 0, "Should have elements"
    assert result['has_form'] == True, "Should detect form elements"
    assert result['has_button'] == True, "Should detect buttons"
    
    # Check specific elements
    elements = result['elements']
    button_elements = [e for e in elements if e.get('tag') == 'button']
    input_elements = [e for e in elements if e.get('tag') == 'input']
    
    print(f"Buttons found: {len(button_elements)}")
    print(f"Input elements found: {len(input_elements)}")
    
    assert len(button_elements) >= 1, "Should find button elements"
    assert len(input_elements) >= 1, "Should find input elements"
    
    # Check element parsing
    first_button = button_elements[0]
    print(f"First button: {first_button}")
    assert first_button['id'] == 1, "Should parse element ID correctly"
    assert first_button['text'] == 'Search', "Should parse text content"
    assert first_button['clickable'] == True, "Should detect clickable attribute"
    
    print("✅ Accessibility tree processing tests passed!")


def test_error_processing():
    """Test processing of error observations"""
    print("\nTesting error processing...")
    
    processor = ObservationProcessor()
    
    error_obs = "Error: Page not found (404)"
    result = processor.process(error_obs)
    
    print(f"Error result: {result}")
    
    assert result['type'] == 'error', "Should be error type"
    assert result['error_message'] == error_obs, "Should preserve error message"
    assert result['num_elements'] == 0, "Should have no elements"
    assert len(result['elements']) == 0, "Elements list should be empty"
    
    print("✅ Error processing tests passed!")


def test_element_parsing():
    """Test individual element line parsing"""
    print("\nTesting element line parsing...")
    
    processor = ObservationProcessor()
    
    # Test various element formats
    test_lines = [
        '[42] <button text="Click Me" clickable>',
        '[100] <input type="password" text="" placeholder="Password">',
        '[5] <div text="Some content here">',
        '[999] <a href="/home" text="Home Page" clickable>',
        '[1] <select type="dropdown">',
        'invalid line without proper format'
    ]
    
    for line in test_lines:
        element = processor._parse_element_line(line)
        print(f"Line: {line}")
        print(f"Parsed: {element}")
        
        if '[42]' in line:
            assert element['id'] == 42, "Should parse ID 42"
            assert element['tag'] == 'button', "Should identify button tag"
            assert element['text'] == 'Click Me', "Should parse text content"
            assert element['clickable'] == True, "Should detect clickable"
        
        if '[100]' in line:
            assert element['id'] == 100, "Should parse ID 100"
            assert element['tag'] == 'input', "Should identify input tag"
            
        if 'invalid line' in line:
            # Should handle malformed lines gracefully
            pass
    
    print("✅ Element parsing tests passed!")


def test_form_detection():
    """Test form and button detection logic"""
    print("\nTesting form and button detection...")
    
    processor = ObservationProcessor()
    
    # Test with form elements
    form_obs = """[1] <form>
[2] <input type="text" text="">
[3] <textarea placeholder="Message">
[4] <select>
[5] <button text="Submit">"""
    
    result = processor.process(form_obs)
    assert result['has_form'] == True, "Should detect form elements"
    assert result['has_button'] == True, "Should detect button"
    
    # Test without form elements
    no_form_obs = """[1] <div text="Just some text">
[2] <p text="Paragraph content">
[3] <span text="More text">"""
    
    result2 = processor.process(no_form_obs)
    assert result2['has_form'] == False, "Should not detect form elements"
    assert result2['has_button'] == False, "Should not detect buttons"
    
    # Test button detection in text
    button_text_obs = """[1] <div text="Click the login button below">
[2] <a text="button link" clickable>"""
    
    result3 = processor.process(button_text_obs)
    assert result3['has_button'] == True, "Should detect button in text content"
    
    print("✅ Form and button detection tests passed!")


if __name__ == "__main__":
    test_accessibility_tree_processing()
    test_error_processing()
    test_element_parsing()
    test_form_detection()
    print("\n🎉 All ObservationProcessor tests passed successfully!")