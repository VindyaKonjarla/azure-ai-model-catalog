dashboard_content = """
<!DOCTYPE html>
<html>
<head>
    <style>
        table {
            border-collapse: collapse;
            width: 100%;
        }

        th, td {
            border: 1px solid #ddd;
            text-align: center;
            padding: 8px;
        }

        th {
            background-color: #f2f2f2;
        }
    </style>
</head>
<body>

<table>
    <tr>
        <th>Model Name</th>
        <th>Status</th>
    </tr>
    <tr>
        <td>Model 1</td>
        <td>In Progress</td>
    </tr>
    <tr>
        <td>Model 2</td>
        <td>Passed</td>
    </tr>
    <tr>
        <td>Model 3</td>
        <td>Failed</td>
    </tr>
</table>

</body>
</html>
"""

# Save the generated HTML content to a file
with open('multimodels_dashboard.md', 'w') as file:
    file.write(dashboard_content)
