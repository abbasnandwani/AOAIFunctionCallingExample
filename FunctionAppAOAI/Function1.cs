using System;
using System.IO;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Azure.WebJobs;
using Microsoft.Azure.WebJobs.Extensions.Http;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using Newtonsoft.Json;
using Azure.AI.OpenAI;
using Azure;
using System.Text.Json;
using System.Collections.Generic;

//https://learn.microsoft.com/en-us/dotnet/api/overview/azure/ai.openai-readme?view=azure-dotnet-preview
//https://dev.to/kenakamu/c-azure-open-ai-and-function-calling-h7j
//https://techcommunity.microsoft.com/t5/ai-azure-ai-services-blog/function-calling-is-now-available-in-azure-openai-service/ba-p/3879241

namespace FunctionAppAOAI
{
    public static class Function1
    {

        private static List<ChatMessage> _ConversationHistory = new List<ChatMessage>();

        [FunctionName("Function1")]
        public static async Task<IActionResult> Run(
            [HttpTrigger(AuthorizationLevel.Anonymous, "get", "post", Route = null)] HttpRequest req,
            ILogger log)
        {
            log.LogInformation("C# HTTP trigger function processed a request.");

            string ques = req.Query["ques"];
            string preserve = req.Query["preserve"];

            string requestBody = await new StreamReader(req.Body).ReadToEndAsync();
            dynamic data = JsonConvert.DeserializeObject(requestBody);
            ques = ques ?? data?.ques;
            preserve = preserve ?? data?.preserve;

            if (!string.Equals(preserve, "true"))
            {
                _ConversationHistory = new List<ChatMessage>();
            }

            string aoaiEndopoint = Environment.GetEnvironmentVariable("aoaiEndopoint");
            Uri aoaiUri = new Uri(aoaiEndopoint);
            string aoaiKey = Environment.GetEnvironmentVariable("aoaiKey");
            string deploymentName = Environment.GetEnvironmentVariable("deploymentName");

            OpenAIClient aoaiClient = new OpenAIClient(aoaiUri,
                new AzureKeyCredential(aoaiKey));

            ChatCompletionsOptions chatCompletionoptions = new ChatCompletionsOptions()
            {
                DeploymentName = deploymentName
            };

            //add historical messages
            foreach (var chatMsg in _ConversationHistory)
            {
                chatCompletionoptions.Messages.Add(chatMsg);
            }


            chatCompletionoptions.Messages.Add(new ChatMessage(ChatRole.User, ques)); //add user question/questions


            //add function definitions
            chatCompletionoptions.Functions.Add(new OpenAIFunctions().GetWeatherFunctionDefinition());
            chatCompletionoptions.Functions.Add(new OpenAIFunctions().GetAddTwoNumbersFunctionDefinition());
            chatCompletionoptions.Functions.Add(new OpenAIFunctions().GetCapitalFunctionDefinition());
            
            Response<ChatCompletions> aoaiResponse = aoaiClient.GetChatCompletions(chatCompletionoptions);

            ChatChoice responseChoice = aoaiResponse.Value.Choices[0];

            while (responseChoice.FinishReason == CompletionsFinishReason.FunctionCall)
            {
                chatCompletionoptions.Messages.Add(responseChoice.Message);

                //execute function
                if (responseChoice.Message.FunctionCall.Name == "get_current_weather")
                {
                    string unvalidatedArguments = responseChoice.Message.FunctionCall.Arguments;

                    //deserialize arguments
                    WeatherInput input = System.Text.Json.JsonSerializer.Deserialize<WeatherInput>(unvalidatedArguments,
                    new JsonSerializerOptions() { PropertyNamingPolicy = JsonNamingPolicy.CamelCase });

                    var functionResultData = new OpenAIFunctions().GetWeather(input.Location, input.Unit);


                    var functionResponseMessage = new ChatMessage(ChatRole.Function,
                        System.Text.Json.JsonSerializer.Serialize(
                                functionResultData,
                                new JsonSerializerOptions() { PropertyNamingPolicy = JsonNamingPolicy.CamelCase }))
                    {
                        Name = responseChoice.Message.FunctionCall.Name
                    };

                    chatCompletionoptions.Messages.Add(functionResponseMessage);
                }
                else if(responseChoice.Message.FunctionCall.Name == "add_two_numbers")
                {
                    string unvalidatedArguments = responseChoice.Message.FunctionCall.Arguments;

                    //deserialize arguments
                    NumberAddition input = System.Text.Json.JsonSerializer.Deserialize<NumberAddition>(unvalidatedArguments,
                    new JsonSerializerOptions() { PropertyNamingPolicy = JsonNamingPolicy.CamelCase });

                    var functionResultData = new OpenAIFunctions().AddTwoNumbers(input.Number1, input.Number2);


                    var functionResponseMessage = new ChatMessage(ChatRole.Function,
                        System.Text.Json.JsonSerializer.Serialize(
                                functionResultData,
                                new JsonSerializerOptions() { PropertyNamingPolicy = JsonNamingPolicy.CamelCase }))
                    {
                        Name = responseChoice.Message.FunctionCall.Name
                    };

                    chatCompletionoptions.Messages.Add(functionResponseMessage);
                }
                else if (responseChoice.Message.FunctionCall.Name == "get_capital")
                {
                    string unvalidatedArguments = responseChoice.Message.FunctionCall.Arguments;

                    //deserialize arguments
                    CapitalInput input = System.Text.Json.JsonSerializer.Deserialize<CapitalInput>(unvalidatedArguments,
                    new JsonSerializerOptions() { PropertyNamingPolicy = JsonNamingPolicy.CamelCase });

                    var functionResultData = new OpenAIFunctions().GetCapital(input.Location);


                    var functionResponseMessage = new ChatMessage(ChatRole.Function,
                        System.Text.Json.JsonSerializer.Serialize(
                                functionResultData,
                                new JsonSerializerOptions() { PropertyNamingPolicy = JsonNamingPolicy.CamelCase }))
                    {
                        Name = responseChoice.Message.FunctionCall.Name
                    };

                    chatCompletionoptions.Messages.Add(functionResponseMessage);
                }


                //call aoai again
                aoaiResponse = aoaiClient.GetChatCompletions(chatCompletionoptions);

                responseChoice = aoaiResponse.Value.Choices[0];
            }

            //preserve history if requested
            if (string.Equals(preserve, "true"))
            {
                foreach (var chatMsg in chatCompletionoptions.Messages)
                {
                    _ConversationHistory.Add(chatMsg);
                }

                _ConversationHistory.Add(responseChoice.Message);
            }

            string responseMessage = responseChoice.Message.Content;

            return new OkObjectResult(responseMessage);
        }
    }


    public class OpenAIFunctions
    {
        #region Weather
        public string Name = "get_current_weather";

        public FunctionDefinition GetWeatherFunctionDefinition()
        {
            return new FunctionDefinition()
            {
                Name = Name,
                Description = "Get the current weather in a given location",
                Parameters = BinaryData.FromObjectAsJson(
                new
                {
                    Type = "object",
                    Properties = new
                    {
                        Location = new
                        {
                            Type = "string",
                            Description = "The city and state, e.g. San Francisco, CA",
                        },
                        Unit = new
                        {
                            Type = "string",
                            Enum = new[] { "Celsius", "Fahrenheit" },
                        }
                    },
                    Required = new[] { "location" },
                },
                new JsonSerializerOptions() { PropertyNamingPolicy = JsonNamingPolicy.CamelCase }),
            };
        }

        public Weather GetWeather(string location, string unit)
        {
            return new Weather() { Temperature = new Random().Next(-4, 40), Unit = unit };
        }

        #endregion

        #region AddTwoNumbers
        public string Name_2 = "add_two_numbers";

        public FunctionDefinition GetAddTwoNumbersFunctionDefinition()
        {
            return new FunctionDefinition()
            {
                Name = Name_2,
                Description = "Adds two numbers",
                Parameters = BinaryData.FromObjectAsJson(
                                   new
                                   {
                                       Type = "object",
                                       Properties = new
                                       {
                                           Number1 = new
                                           {
                                               Type = "number",
                                               Description = "The first number",
                                           },
                                           Number2 = new
                                           {
                                               Type = "number",
                                               Description = "The second number",
                                           }
                                       },
                                       Required = new[] { "number1", "number2" },
                                   },
                                                  new JsonSerializerOptions() { PropertyNamingPolicy = JsonNamingPolicy.CamelCase }),
            };
        }

        public double AddTwoNumbers(double number1, double number2)
        {
            return number1 + number2;
        }
        #endregion


        #region Get Capital
        public string Name_3 = "get_capital";
        
        public FunctionDefinition GetCapitalFunctionDefinition()
        {
            return new FunctionDefinition()
            {
                Name = Name_3,
                Description = "Get the capital of the location",
                Parameters = BinaryData.FromObjectAsJson(
                new
                {
                    Type = "object",
                    Properties = new
                    {
                        Location = new
                        {
                            Type = "string",
                            Description = "The city, state or country, e.g. San Francisco, CA",
                        }
                    },
                    Required = new[] { "location" },
                },
                new JsonSerializerOptions() { PropertyNamingPolicy = JsonNamingPolicy.CamelCase }),
            };
        }

        // The function implementation. It always return Tokyo for now.
        public string GetCapital(string location)
        {

            switch (location.ToLower())
            {
                case "japan":
                    return "Tokyo";

                case "uk":
                    return "London";

                case "usa":
                    return "Washington DC";

                default:
                    return "Unknown";
            }

        }
        #endregion
    }

    // Argument for the function
    public class WeatherInput
    {
        public string Location { get; set; } = string.Empty;
        public string Unit { get; set; } = "Celsius";
    }

    // Return type
    public class Weather
    {
        public int Temperature { get; set; }
        public string Unit { get; set; } = "Celsius";
    }

    public class NumberAddition
    {
        public double Number1 { get; set; }
        public double Number2 { get; set; }
    }

    public class CapitalInput
    {
        public string Location { get; set; } = string.Empty;
    }
}
