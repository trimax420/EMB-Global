import AWS from 'aws-sdk';

export const configureAWS = () => {
  AWS.config.update({
    region: process.env.AWS_REGION,
    accessKeyId: process.env.AWS_ACCESS_KEY_ID,
    secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY,
  });

  return AWS;
};

export default configureAWS;
