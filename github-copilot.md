## GitHub Copilot Usage Guide

This guide provides tips for using GitHub Copilot to its full potential in the Whisper Transcriber project.

### Code Completion

Copilot excels at autocompleting code. As you type, it will suggest completions for lines or entire functions. For example, when writing a Flask route, you can start with:

```python
@app.route('/status/<job_id>')
def get_status(job_id):
```

Copilot will likely suggest the entire implementation based on the existing code patterns.

### Writing Repetitive Code

Use Copilot to generate boilerplate or repetitive code. For example, if you are adding a new option to the settings UI, Copilot can help generate the HTML and JavaScript needed to handle the new option.

### Inline Instructions

You can write a comment describing the function you want to create, and Copilot will often generate the code for you:

```python
# function to format seconds into HH:MM:SS.ms format
```

### Limitations

- **Context is Key:** Copilot's suggestions are based on the current file and open tabs. Ensure you have relevant files open to get the best suggestions.
- **Review Suggestions:** Always review Copilot's suggestions carefully. They may not always be correct or optimal.
