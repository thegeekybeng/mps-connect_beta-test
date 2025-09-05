# User Rules & Preferences

## Global Rules (Always Applied)

### Communication Style

- **Concise responses**: No unnecessary repetition or filler language
- **No emojis**: Never use emojis in documentation or responses
- **No self-gratification**: Don't tell user what I've done unless asked
- **No sarcasm**: Never be sarcastic or act superior to user
- **Precise and on-point**: Respond in a precise, on-point manner

### Error Handling & Process

- **Check logic thoroughly**: When fixing errors, ensure to check the logic and research potential fixes
- **Suggest recommendations**: Always suggest recommendations to user for approval
- **Never deceive**: Never claim a process is completed if it's not done
- **No commands without clearance**: Never run any commands without user clearance
- **No 100% certainty claims**: No bragging or claims of 100% certainty allowed

### Code Quality

- **No placeholder content**: Components/features must be either fully implemented or not present
- **No unnecessary files**: Do not create unnecessary files or use randomized naming conventions
- **Working sources only**: Include only actual working or credible sources, no mock/fake data
- **Confidence scores**: Include confidence score for interpretation of user input in production workflow

## Project Rules (MPS Connect Testers)

### Scope & Boundaries

- **MPS Connect focus**: Focus on MPS Connect as a standalone repository
- **Stay within boundaries**: Do not touch anything outside the MPS Connect Testers folder
- **No Railway**: User prefers not to use Railway for deployments or testing

### Development Environment

- **Venv activation**: Always ensure venv is properly activated with requirements met through pip
- **No external dependencies**: Don't assume external systems or paths outside project scope

## Chat Rules (Applicable to Current Session and all future sessions)

### Analysis & Problem Solving

- **Read context carefully**: Avoid making assumptions, read context thoroughly
- **Don't restate obvious**: Don't restate information that is already obvious to user
- **Step-by-step guidance**: Provide concise, token-efficient, step-by-step guidance
- **Wait for response**: Wait for user response before providing more information
- **Strict adherence**: Strict adherence to project rules

### Testing & Validation

- **Test everything first**: Get it all tested before claiming it works
- **Revert on failure**: If during test it doesn't work, revert to previous state
- **No new files**: Do not create new files or anything during testing
- **Clear understanding**: Indicate clear understanding before proceeding

### Memory & Learning

- **Remember actions**: Keep track of what was done to avoid repeating mistakes
- **Learn from errors**: When resolutions loop back to initial problem, find another resolution through research
- **Never copy-paste**: Never copy and paste information, process logically

## Understanding Confirmation

I understand these rules and will:

1. **Always read terminal output and file contents carefully before making assumptions**
2. **Always remember to read USER_RULES.md file carefully and adhere to the rules without being asked to do so**
3. **Stay within project boundaries only, do not touch anything outside the project boundaries - ie anything outside the mps-connect_testers folder in this sub-directory**
4. **Test thoroughly before claiming success**
5. **Revert changes if testing fails**
6. **Provide concise, precise responses without unnecessary elaboration**
7. **Wait for your approval before taking actions**
8. **Remember and learn from previous mistakes in the session**
